import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from .utils.self_attention import *
from .utils.layernorm import *
from .utils.activation import * 
from .utils.linear import *
from .utils.modules import *
import pickle

#This module stacks the necessary TinT modules for a transformer block
#Arguments:
#config, \                       #TinT config file
#model_config, \                 #config file of Auxiliary model
#block, \                        #auxiliary transformer's block
#add_biases_prefixes, \          #If true, we add new variables as bias terms that contain the auxiliary model's parameters  
#ln_memory_index, \              #memory index where we store the layernorm activations (for backpropagation) during forward pass
#attn_memory_index, \            #memory index where we store the self_attention activations (for backpropagation) during forward pass
#linear_memory_index, \          #memory index where we store the linear activations (for backpropagation) during forward pass
#act_memory_index, \             #memory index where we store the activation layer activations (for backpropagation) during forward pass
#layer_id,\                      #index number of the transformer layer 
#separate_QK=False, \            #True/False, separate operations for query and key computation
#project_MLP=False, \            #True/False, whether to project the weights of MLP
class optblock_forward(nn.Module):
    
    def __init__ (self, \
                  config, \
                  model_config, \
                  block, \
                  add_biases_prefixes, \
                  ln_memory_index, \
                  attn_memory_index, \
                  linear_memory_index, \
                  act_memory_index, \
                  layer_id,\
                  separate_QK=False, \
                  project_MLP=False, \
                 ):
        super(optblock_forward, self).__init__()
        self.config=config
        self.add_biases_prefixes = add_biases_prefixes
        self.ln_memory_index = ln_memory_index
        self.attn_memory_index = attn_memory_index
        self.linear_memory_index = linear_memory_index 
        self.act_memory_index = act_memory_index
        self.separate_QK = separate_QK
        self.memory_locations = []
        self.trainable_biases = []
        self.layer_id = layer_id
        self.project_MLP = project_MLP
        self.reqd_biases = []
        
        
        ############## -------------------------- Attention module -------------------------- ##############
        #Modules for forward through attention module
        
        ln_1 = LayerNormForward(config, \
                                din=model_config.hidden_size, \
                                use_softmax=False, \
                                memory_index=self.ln_memory_index \
                               )
        self.add_module('attention_ln', ln_1)
        self.memory_locations += [self.ln_memory_index]
        if  add_biases_prefixes: self.add_biases([ block['self_attn_layer_norm'] ], diagonal=True )
        
        #Create a projection matrix for Query and Key matrices concatenated if separate_QK=True
        if separate_QK:
            w = torch.cat( [block['self_attn']['q_proj'].weight, block['self_attn']['k_proj'].weight, block['self_attn']['v_proj'].weight], dim=0 )
            QK, _ = torch.split(w, split_size_or_sections=[ 2 * w.shape[0] // 3, w.shape[0] // 3 ], dim=0)
            projection_matrix_QK,  _, _ = np.linalg.svd( QK.detach().cpu().numpy(), full_matrices=False, compute_uv=True )
            projection_matrix = projection_matrix_QK
        else:
            projection_matrix = None
        
        attnt = AttentionForward(config, \
                                 din=model_config.hidden_size, \
                                 num_attnt_heads=model_config.num_attention_heads, \
                                 use_softmax=False, \
                                 separate_QK=separate_QK, \
                                 memory_index=self.attn_memory_index, \
                                 projection_matrix=projection_matrix, \
                                )
        self.add_module('attention', attnt)
        self.memory_locations += [self.attn_memory_index]
        
        
        if  add_biases_prefixes: 
            wt_projection = projection_matrix.T if projection_matrix is not None else None
            self.add_biases([ block['self_attn']['q_proj'], block['self_attn']['k_proj'], block['self_attn']['v_proj'] ], separate_QK=separate_QK, attention=True, projection=wt_projection)         
                
        attnt_proj = LinearForward(config, \
                                   din=model_config.hidden_size, \
                                   dout=model_config.hidden_size, \
                                   use_softmax=False, \
                                   memory_index=self.linear_memory_index, \
                                  )
        
        self.add_module('attention_projection', attnt_proj)
        self.memory_locations += [self.linear_memory_index]
        if  add_biases_prefixes: self.add_biases([ block['self_attn']['out_proj'] ])
                
        self.attnt_modules = [ln_1, attnt, attnt_proj]
        
        ############## -------------------------------------------------------------------------- ##############
        
        
        ############## ------------------------------- MLP module ------------------------------- ##############
        #Modules for forward through mlp module
        ln_2 = LayerNormForward(config, \
                                din=model_config.hidden_size, \
                                use_softmax=False, \
                                memory_index=self.ln_memory_index, \
                               )
        
        self.add_module('ln_mlp', ln_2)
        self.memory_locations += [self.ln_memory_index]
        if  add_biases_prefixes: self.add_biases([ block['final_layer_norm'] ], diagonal=True)
        
        self.mlp_modules = [ln_2]
            
        inner_dim=model_config.ffn_dim if model_config.ffn_dim is not None else 4 * model_config.hidden_size
        
        
        self.inner_projection_matrix_wts = []
        self.inner_projection_matrixes = []

        self.outer_projection_matrix_wts = []
        self.outer_projection_matrixes = []    
        
        if project_MLP:
            #Create a projection matrix for the MLP hidden layer if project_MLP=True
            mlp_proj_inner_wt, \
            mlp_proj_inner, \
            mlp_proj_outer, \
            mlp_proj_outer_wt = pickle.load(open(self.config.projection_paths + '/projection_'+str(layer_id)+'.pkl', 'rb'))
            
            #wt = block['mlp']['c_fc'].weight.T.detach().cpu().numpy()
            inner_projection_matrix_wt = mlp_proj_inner_wt
            inner_projection_matrix = mlp_proj_inner
            
            outer_projection_matrix_wt  = mlp_proj_outer_wt
            outer_projection_matrix = mlp_proj_outer
            #,  _, _ = np.linalg.svd( wt, full_matrices=False, compute_uv=True )
            
            
            self.inner_projection_matrix_wts += [inner_projection_matrix_wt]
            self.inner_projection_matrixes += [inner_projection_matrix]

            self.outer_projection_matrix_wts += [outer_projection_matrix_wt]
            self.outer_projection_matrixes += [outer_projection_matrix]
        
            #wt = block['mlp']['c_proj'].weight.T.detach().cpu().numpy()
            #_,  _, self.outer_projection_matrix = np.linalg.svd( wt, full_matrices=False, compute_uv=True )
            hid_size = model_config.hidden_size
            self.num_mlp_modules = 1
        else:
            self.num_mlp_modules = inner_dim // model_config.hidden_size            
            for k in range(self.num_mlp_modules):
                projection_matrix = np.zeros((inner_dim, model_config.hidden_size))
                projection_matrix[ k*model_config.hidden_size: (k+1)*model_config.hidden_size ] = np.eye(model_config.hidden_size)
                self.inner_projection_matrix_wts += [projection_matrix.T]
                self.inner_projection_matrixes   += [None]
                
                self.outer_projection_matrix_wts += [projection_matrix]
                self.outer_projection_matrixes   += [None]
                
            #self.inner_projection_matrix_wt  = None
            #self.inner_projection_matrix = None
            
            #self.outer_projection_matrix_wt  = None
            #self.outer_projection_matrix = None
            #inner_projection_matrix = None
            hid_size = model_config.hidden_size
        
        
        #for k in range(self.num_mlp_modules):
        intermediate_layer = LinearForward(config, \
                                           din=model_config.hidden_size, \
                                           dout=hid_size, \
                                           use_softmax=False, \
                                           memory_index=self.linear_memory_index, \
                                           projection_matrix=self.inner_projection_matrixes[0], \
                                          )
        self.add_module('intermediate_mlp', intermediate_layer)
        #self.memory_locations += [self.linear_memory_index]
        #if  add_biases_prefixes: 
            


        #for k in range(self.num_mlp_modules):
        act_din = inner_dim if self.project_MLP else hid_size
        activation_layer = ActivationForward(config, \
                                             din=act_din, \
                                             memory_index=self.act_memory_index, \
                                             projection_matrix=self.outer_projection_matrixes[0], \
                                            )
        self.add_module('activation_mlp', activation_layer)
        #self.memory_locations += [self.act_memory_index]
        #if  add_biases_prefixes: 
            


        output_layer = LinearForward(config, 
                                     din=hid_size, \
                                     dout=model_config.hidden_size, 
                                     use_softmax=False,\
                                     memory_index=self.linear_memory_index,
                                    ) 
        self.add_module('mlp_projection', output_layer)
        #self.memory_locations += [self.linear_memory_index]

        if  add_biases_prefixes: 
            for k in range(self.num_mlp_modules):
                wt_projection = self.inner_projection_matrix_wts[k]
                #self.inner_projection_matrix.T if self.inner_projection_matrix is not None else None
                self.add_biases([ block['fc1'] ], projection=wt_projection)      
                
                self.add_biases(all_zeros=True)  
                
                wt_projection = self.outer_projection_matrix_wts[k]
                self.add_biases([ block['fc2'] ], projection=wt_projection, project_input=True, project_bias= self.num_mlp_modules) 
                
        for _ in range(self.num_mlp_modules):
            self.mlp_modules += [intermediate_layer, activation_layer, output_layer] 
            self.memory_locations += [self.linear_memory_index, self.act_memory_index, self.linear_memory_index]
            
            
        self.trainable_biases = nn.ParameterList(self.trainable_biases)
    #This function adds weights of the original model as biases, 
    #that can later be added to the blank tokens.
    def add_biases(self, \
                   tensors=[], \
                   diagonal=False, \
                   all_zeros=False, \
                   projection=None, \
                   project_input=False, \
                   project_bias=1., \
                   separate_QK=False, \
                   attention=False, \
                  ):
        
        biases = nn.Parameter(torch.zeros(self.config.num_prefixes, self.config.hidden_size))
        
        if all_zeros:
            self.trainable_biases += [biases]
            return
 
        #w = tensor.weight.T
        #b = tensor.bias
        w = torch.cat([tensor.weight for tensor in tensors], dim=0)
        b = torch.cat([tensor.bias for tensor in tensors], dim=0)
        
        
        if separate_QK:
            w, V = torch.split(w, split_size_or_sections=[ 2*w.shape[0] // 3, w.shape[0] // 3 ], dim=0)
            b, V_b = torch.split(b, split_size_or_sections=[ 2*b.shape[0] // 3, b.shape[0] // 3 ], dim=0)
        elif attention:
            w, K, V = torch.split(w, split_size_or_sections=[ w.shape[0] // 3, w.shape[0] // 3, w.shape[0] // 3 ], dim=0)
            b, K_b, V_b = torch.split(b, split_size_or_sections=[ b.shape[0] // 3, b.shape[0] // 3, b.shape[0] // 3 ], dim=0)
        
        if projection is not None:
            projection_tensor = torch.tensor( projection, dtype=w.dtype ).to(w.device)
            if project_input:
                w = w @ projection_tensor
                b = b / project_bias
            else:
                w = projection_tensor @ w
                b = projection_tensor @ b
            
        num_wts_per_blank=w.shape[0] // self.config.num_prefixes 
        if not diagonal:
            din = w.shape[1]
        else:
            din = w.shape[0]
            
            
        with torch.no_grad():
            if diagonal: reshaped_w = torch.diag(w)
            else: reshaped_w = w
            reshaped_w = reshaped_w.reshape(self.config.num_prefixes, num_wts_per_blank * din)
            biases[: , : num_wts_per_blank * din ] = reshaped_w
            biases[: , num_wts_per_blank * din : num_wts_per_blank * din + din ] = b        
        self.trainable_biases += [biases]
        
        #if separate_QK:
        if not separate_QK and attention:
            new_biases = nn.Parameter(torch.zeros(self.config.num_prefixes, self.config.hidden_size))
            w = K
            b = K_b

            num_wts_per_blank=w.shape[0] // self.config.num_prefixes 
            din = w.shape[1]

            with torch.no_grad():
                if diagonal: reshaped_w = torch.diag(w)
                else: reshaped_w = w
                reshaped_w = reshaped_w.reshape(self.config.num_prefixes, num_wts_per_blank * din)
                new_biases[: , : num_wts_per_blank * din ] = reshaped_w
                new_biases[: , num_wts_per_blank * din : num_wts_per_blank * din + din ] = b        
            self.trainable_biases += [new_biases]

        if attention:
            new_biases = nn.Parameter(torch.zeros(self.config.num_prefixes, self.config.hidden_size))
            w = V
            b = V_b

            num_wts_per_blank=w.shape[0] // self.config.num_prefixes 
            din = w.shape[1]

            with torch.no_grad():
                if diagonal: reshaped_w = torch.diag(w)
                else: reshaped_w = w
                reshaped_w = reshaped_w.reshape(self.config.num_prefixes, num_wts_per_blank * din)
                new_biases[: , : num_wts_per_blank * din ] = reshaped_w
                new_biases[: , num_wts_per_blank * din : num_wts_per_blank * din + din ] = b        
            self.trainable_biases += [new_biases]

        
    def forward(self, \
                hidden_states, \
                position_embeddings, \
                past_memory_prefixes=None, \
                initial_forward=True, \
                icl_mask=None, \
               ):
        
        counter = 0
        memory_activation_counter = 0
        residual = hidden_states
        memory_prefixes = []
        memory_activations = []
   
        
        assert len(self.trainable_biases) > 0 or past_memory_prefixes is not None, \
               "Need weights in the prefixes anyhow!"
        
        out_hidden_states = hidden_states
        
        
        #print ("---------attention----------")   
        for module in self.attnt_modules:
            
            
            hidden_states = out_hidden_states
            
            if initial_forward and len(self.trainable_biases) > 0:
                #hidden_states[:, :self.config.num_prefixes] += ( self.trainable_biases[counter] - hidden_states[:, :self.config.num_prefixes] )
                hidden_states = torch.cat( [ self.trainable_biases[counter].unsqueeze(0).expand(hidden_states[:, :self.config.num_prefixes].shape), hidden_states[:, self.config.num_prefixes:]  ], axis=1 )
            else:
                hidden_states = torch.cat( [ past_memory_prefixes[counter], hidden_states[:, self.config.num_prefixes:]  ], axis=1 )
                #hidden_states[:, :self.config.num_prefixes] += ( past_memory_prefixes[counter] - hidden_states[:, :self.config.num_prefixes] )
            
            
            counter += 1
            #store the weights for the future backward passes
            memory_prefixes += [hidden_states[:, :self.config.num_prefixes]]
            
            
            
            if counter == 2:
                if not self.separate_QK:
                    if initial_forward  and len(self.trainable_biases) > 0:   
                        key_wts = self.trainable_biases[counter].unsqueeze(0).expand(memory_prefixes[-1].shape)
                    else:
                        key_wts = past_memory_prefixes[counter]
                    #store the weights for the future backward passes
                    memory_prefixes += [ key_wts ]
                    counter += 1    
                else:
                    key_wts = None
                    
                if initial_forward  and len(self.trainable_biases) > 0:   
                    value_wts = self.trainable_biases[counter].unsqueeze(0).expand(memory_prefixes[-1].shape)
                else:
                    value_wts = past_memory_prefixes[counter]
                #store the weights for the future backward passes
                memory_prefixes += [ value_wts ]
                counter += 1    
                
                
                #memory_activations += [ out_hidden_states[:, self.config.num_prefixes:, mem_loc:] ]
                #push the first din coordinates of input into memory
                #memory_activations += [ hidden_states[:, self.config.num_prefixes:, :self.model_config.hidden_size] ]
                
                out_hidden_states = module (hidden_states, \
                                            position_embeddings, \
                                            key_weights=key_wts, \
                                            value_weights=value_wts, \
                                            icl_mask=icl_mask, \
                                           ) 
                
            else:
                out_hidden_states = module (hidden_states, position_embeddings) 
                
            #store the memory for the future backward passes
            mem_loc = self.memory_locations[memory_activation_counter]
            memory_activations += [ out_hidden_states[:, self.config.num_prefixes:, mem_loc:] ]
            memory_activation_counter += 1
            
            #print (module)
            
        #print ("---------mlp----------")       
            
        hidden_states = out_hidden_states + residual
               
        residual = hidden_states
        out_hidden_states = hidden_states
        module_counter = 0
        
        #print (self.mlp_modules)
        for module in self.mlp_modules:
            
            
            if module_counter == 1:
                chunk_hidden_states = out_hidden_states
            
            if (module_counter - 1) % 3 == 0:
                hidden_states = chunk_hidden_states
            else:
                hidden_states = out_hidden_states
                
            #if module_counter == 1:
            #    chunk_hidden_states = [out_hidden_states, out_hidden_states, out_hidden_states, out_hidden_states]
            
            #if module_counter >= 1:
                #for k in range(self.num_mlp_modules):
                
            if initial_forward and len(self.trainable_biases) > 0:
                #hidden_states[:, :self.config.num_prefixes] += ( self.trainable_biases[counter] - hidden_states[:, :self.config.num_prefixes] )
                hidden_states = torch.cat( [ self.trainable_biases[counter].unsqueeze(0).expand(hidden_states[:, :self.config.num_prefixes].shape), hidden_states[:, self.config.num_prefixes:]  ], axis=1 )
                #hidden_states = torch.cat( [ self.trainable_biases[counter], hidden_states[:, self.config.num_prefixes:]  ], axis=1 )
            else:
                hidden_states = torch.cat( [ past_memory_prefixes[counter], hidden_states[:, self.config.num_prefixes:]  ], axis=1 )
                #hidden_states[:, :self.config.num_prefixes] += ( past_memory_prefixes[counter] - hidden_states[:, :self.config.num_prefixes] )

            counter += 1
            #store the weights for the future backward passes
            memory_prefixes += [hidden_states[:, :self.config.num_prefixes]]

            out_hidden_states = module (hidden_states, position_embeddings)
            #store the memory for the future backward passes
            mem_loc =  self.memory_locations[memory_activation_counter]
            memory_activations += [ out_hidden_states[:, self.config.num_prefixes:, mem_loc:] ]
            memory_activation_counter += 1
            
            module_counter += 1  
            
            
            if module_counter == 4:
                final_state = out_hidden_states
            elif module_counter > 4 and (module_counter-4) % 3 == 0:
                final_state = final_state + out_hidden_states
            
            #print (module)
        
        
        hidden_states = final_state + residual
        return hidden_states, memory_prefixes, memory_activations
        
#This module stacks the necessary TinT modules for the backpropagation through a transformer block
#Arguments:
#config, \                       #TinT config file
#model_config, \                 #config file of Auxiliary model
#ln_memory_index, \              #memory index where we store the layernorm activations (for backpropagation) during forward pass
#attn_memory_index, \            #memory index where we store the self_attention activations (for backpropagation) during forward pass
#linear_memory_index, \          #memory index where we store the linear activations (for backpropagation) during forward pass
#act_memory_index, \             #memory index where we store the activation layer activations (for backpropagation) during forward pass
#separate_QK=False, \            #True/False, separate operations for query and key computation
#project_MLP=False, \            #True/False, whether to project the weights of MLP
#mlp_inner_projection_matrix     #Only needed when we apply projections to the linear operations of MLP, currently unnecesary 
#mlp_outer_projection_matrix     #Only needed when we apply projections to the linear operations of MLP, currently unnecesary 
#memory_locations                #contains the memory indices for all the operations in the transformer block 
        
class optblock_backward(nn.Module):
    
    
    def __init__(self, \
                 config, \
                 model_config, \
                 ln_memory_index, \
                 attn_memory_index, \
                 linear_memory_index, \
                 act_memory_index, \
                 separate_QK=False, \
                 project_MLP=False, \
                 mlp_inner_projection_matrix=None, \
                 mlp_outer_projection_matrix=None, \
                 memory_locations=[], \
                ):
        
        super(optblock_backward, self).__init__()
        assert not project_MLP or ( mlp_inner_projection_matrix is not None and mlp_outer_projection_matrix is not None ), \
               "Either no projection in MLP or mlp projection matrices are provided!"
        
        self.ln_memory_index = ln_memory_index
        self.attn_memory_index = attn_memory_index
        self.linear_memory_index = linear_memory_index 
        self.act_memory_index = act_memory_index
        self.separate_QK = separate_QK
        self.memory_locations = memory_locations
        self.project_MLP = project_MLP
        self.config = config
        
        #self.update_memory = []
        #self.store_memory = []
        self.trainable_biases = []
        
        #Modules for forward through mlp module
        inner_dim=model_config.ffn_dim if model_config.ffn_dim is not None else 4 * model_config.hidden_size
        #if self.project_MLP: mlp_innerdim = model_config.hidden_size
        #else: mlp_innerdim = inner_dim
        mlp_innerdim = model_config.hidden_size
        self.num_mlp_modules = inner_dim // mlp_innerdim if not project_MLP else 1
        
        #output_backward = LinearBackward(config, \
        #                                 din=mlp_innerdim, \
        #                                 dout=model_config.hidden_size, \
        #                                 use_softmax=False, \
        #                                 memory_index=self.linear_memory_index, \
        #                                )
        #self.add_module('mlp_projection_back', output_backward)
        
        output_descent  = Linear_Descent_Backward(config, \
                                                  din=mlp_innerdim, \
                                                  dout=model_config.hidden_size, \
                                                  use_softmax=False, \
                                                  memory_index=self.linear_memory_index, \
                                                  debug_zero=False, \
                                                 )
        self.add_module('mlp_projection_back_descent', output_descent)
        
        wt_projection = mlp_outer_projection_matrix.T if mlp_outer_projection_matrix is not None else None
        act_projection=mlp_inner_projection_matrix.T if mlp_inner_projection_matrix is not None else None
        
        act_din=inner_dim if self.project_MLP else mlp_innerdim
        activation_backward = ActivationBackward(config, \
                                                 din=act_din, \
                                                 input_projection=wt_projection,\
                                                 projection_matrix=act_projection,\
                                                 memory_index=self.act_memory_index,\
                                                )
        self.add_module('mlp_activation_back', activation_backward)
        
        intermediate_descent = Linear_Descent_Backward(config, \
                                                       din=model_config.hidden_size, \
                                                       dout=mlp_innerdim,\
                                                       use_softmax=False,\
                                                       projection_matrix=mlp_inner_projection_matrix,\
                                                       memory_index=self.linear_memory_index,\
                                                       debug_zero=False
                                                      )
        self.add_module('mlp_intermediate_back_descent', intermediate_descent)
        
        #intermediate_descent = LinearDescent(config, \
        #                                     din=model_config.hidden_size, \
        #                                     dout=mlp_innerdim, \
        #                                     use_softmax=False, \
        #                                     memory_index=self.linear_memory_index,\
        #                                     debug_zero=False
        #                                    )
        #self.add_module('mlp_intermediate_descent', intermediate_descent)
        
        ln_2_backward = LayerNormDescent_Backward(config, \
                                                  din=model_config.hidden_size, \
                                                  use_softmax=False, \
                                                  memory_index=self.ln_memory_index,
                                                 )
        self.add_module('mlp_ln_back', ln_2_backward)
        
        #ln_2_descent  = LayerNormDescent(config, \
        #                                 din=model_config.hidden_size, \
        #                                 use_softmax=False, \
        #                                 memory_index=self.ln_memory_index, \
        #                                 debug_zero=True
        #                                )
        #self.add_module('mlp_ln_descent', ln_2_descent)
        
        self.mlp_modules   = []
        #self.update_memory = []
        self.skip_memory   = []
        #self.store_memory  = []

            
        for _ in range(self.num_mlp_modules):
            self.mlp_modules   += [output_descent, activation_backward, intermediate_descent]
            #self.update_memory += [True, True, True]
            self.skip_memory   += [False, False, False]
            #self.store_memory  += [True, True, True]
            
        self.num_mlp_repetitive = 3
        self.ln_mlp_module_index = len(self.mlp_modules)
        self.mlp_modules   += [ln_2_backward]
        #self.update_memory += [True]
        self.skip_memory   += [False]
        #self.store_memory  += [True]
        
        #Modules for forward through attention module
        #attnt_proj_backward = LinearBackward(config, \
        #                                     din=model_config.hidden_size, \
        #                                     dout=model_config.hidden_size, \
        #                                     use_softmax=False, \
        #                                     memory_index=self.linear_memory_index, \
        #                                    )
        #self.add_module('attention_projection_back', attnt_proj_backward)
        
        attnt_proj_descent = Linear_Descent_Backward(config, \
                                                     din=model_config.hidden_size, \
                                                     dout=model_config.hidden_size, \
                                                     use_softmax=False, \
                                                     memory_index=self.linear_memory_index, \
                                                     debug_zero=False
                                                    )
        self.add_module('attention_projection_back_descent', attnt_proj_descent)
        
        #attnt_backward = LightAttentionBackward(config, \
        #                                        din=model_config.hidden_size, \
        #                                        num_attnt_heads=model_config.num_attention_heads, \
        #                                        memory_index=self.attn_memory_index, \
        #                                        use_softmax=False, \
        #                                       )
        #self.add_module('attention_back', attnt_backward)
        
        attnt_descent  = AttentionBackward_Descent(config, \
                                                    din=model_config.hidden_size, \
                                                    num_attnt_heads=model_config.num_attention_heads, \
                                                    use_softmax=False, \
                                                    memory_index=self.attn_memory_index, \
                                                    debug_zero=False,\
                                                   )
        self.add_module('attention_back_descent', attnt_descent)
        
        ln_1_backward = LayerNormDescent_Backward(config, \
                                                  din=model_config.hidden_size, \
                                                  use_softmax=False, \
                                                  memory_index=self.ln_memory_index, \
                                                 )
        self.add_module('attention_ln_back', ln_1_backward)
        
        #ln_1_descent  = LayerNormDescent(config, \
        #                                 din=model_config.hidden_size, \
        #                                 use_softmax=False, \
        #                                 memory_index=self.ln_memory_index, \
        #                                 debug_zero=True
        #                                )
        #self.add_module('attention_ln_descent', ln_1_descent)
        
        
        self.attnt_modules= [attnt_proj_descent, attnt_descent, ln_1_backward]
        #self.update_memory += [True, True, True]
        self.skip_memory   += [False, True, False]
        #self.store_memory  += [True, True, True]
        #self.additional_act = [False, True, False]
    
        self.trainable_biases = nn.ParameterList(self.trainable_biases)
    
    def forward(self, \
                hidden_states, \
                position_embeddings, \
                attention_mask, \
                memory_activations, \
                memory_prefixes, \
                icl_mask=None,
               ):
        
        residual = hidden_states
        memory_counter = 0
        activation_counter = len(memory_activations)-1
        weight_counter  = len(memory_prefixes)-1
        new_memory_prefixes = []
        
        out_hidden_states = hidden_states
        module_counter = 0
        
        for module in self.mlp_modules:
            if module_counter == 0:
                chunk_hidden_states = out_hidden_states
                
            if module_counter % self.num_mlp_repetitive == 0 and module_counter < self.ln_mlp_module_index:
                hidden_states = chunk_hidden_states
            else:
                hidden_states = out_hidden_states
            
            
            #if self.update_memory[memory_counter]:
            #first copy the weights onto the blank tokens 
            #old_memory=memory_prefixes[weight_counter]
            #hidden_states[:, :self.config.num_prefixes] += ( memory_prefixes[weight_counter] - hidden_states[:, :self.config.num_prefixes] ) <---- in-place operation

            hidden_states = torch.cat([ memory_prefixes[weight_counter], hidden_states[:, self.config.num_prefixes:] ], axis=1)
            
            #if self.additional_act[memory_counter]:
            #    additional_act = memory_activations[activation_counter]
            #    activation_counter -= 1
            #else:
            #    additional_act = None
            #further copy the memory on activations 
            mem_loc=self.memory_locations[activation_counter]

            hidden_states[:, self.config.num_prefixes:, mem_loc:] += ( memory_activations[activation_counter] - hidden_states[:, self.config.num_prefixes:, mem_loc:] )
            #print (memory_activations[activation_counter][0])
            

            activation_counter -= 1

            weight_counter -= 1
            
            #print ("In", torch.amax(torch.absolute(hidden_states[:, self.config.num_prefixes:, :768])))
            out_hidden_states = module (hidden_states, position_embeddings, attention_mask)
            #print ("Out", torch.amax(torch.absolute(out_hidden_states[:, self.config.num_prefixes:, :768])))
            
            
            #print ( torch.max ( torch.absolute(out_hidden_states[:, :self.config.num_prefixes] - hidden_states[:,:self.config.num_prefixes] ) )  )
            
            #push the new weights into memory
            #if self.store_memory[memory_counter]:
            new_memory=out_hidden_states[:, :self.config.num_prefixes]
            new_memory_prefixes += [new_memory]

            memory_counter += 1
            module_counter += 1
            
            
            if module_counter == self.num_mlp_repetitive:
                final_state = out_hidden_states
            elif module_counter % self.num_mlp_repetitive == 0 and module_counter < self.ln_mlp_module_index:
                final_state = final_state + out_hidden_states
            elif module_counter == self.ln_mlp_module_index:
                out_hidden_states = final_state + out_hidden_states
                
        hidden_states = out_hidden_states + residual
        #print ("-----mlp----")
        
        residual = hidden_states
        stack_QK = None
        out_hidden_states = hidden_states
        
        for module in self.attnt_modules:
            hidden_states = out_hidden_states
            #if self.update_memory[memory_counter]:
            #first copy the weights onto the blank tokens
            #old_memory=memory_prefixes[weight_counter]
            #hidden_states[:, :self.config.num_prefixes] += ( memory_prefixes[weight_counter] - hidden_states[:, :self.config.num_prefixes] ) <---- in-place operation
            hidden_states = torch.cat([ memory_prefixes[weight_counter], hidden_states[:, self.config.num_prefixes:] ], axis=1)
            #further copy the memory on activations 
            mem_loc=self.memory_locations[activation_counter]
            hidden_states[:, self.config.num_prefixes:, mem_loc:] += ( memory_activations[activation_counter] - hidden_states[:, self.config.num_prefixes:, mem_loc:] )
            activation_counter -= 1
            #Skip uploading QK onto the blank tokens
            if self.skip_memory [memory_counter]: 
                weight_counter -= 1
                if not self.separate_QK:
                    #push key matrix
                    stack_QK = [ memory_prefixes[weight_counter] ]
                    weight_counter -= 1
                else:
                    stack_QK = []

                #push query matrix
                stack_QK += [ memory_prefixes[weight_counter] ]
                weight_counter -= 1
            else: 
                weight_counter -= 1

            out_hidden_states = module (hidden_states, position_embeddings, attention_mask, icl_mask=icl_mask) 
            #print (torch.amax(torch.absolute(out_hidden_states[:, self.config.num_prefixes:, :768])))
            
            #push the new weights into memory
            #if self.store_memory[memory_counter]:
            new_memory=out_hidden_states[:, :self.config.num_prefixes]
            new_memory_prefixes += [new_memory]
            if stack_QK is not None:
                for mat in stack_QK: new_memory_prefixes += [mat]
                stack_QK = None

            memory_counter += 1
        #print ("-----attention----")    
   
        hidden_states = out_hidden_states + residual
        
        #reverse the memory in prefixes for future forward passes
        new_memory_prefixes.reverse()
        return hidden_states, new_memory_prefixes
        


#This module applies a layernorm on the final computation and then computes gradient W.R.T. classification error
#Arguments:
#config, \                       #TinT config file
#model_config, \                 #config file of Auxiliary model
#block, \                        #layernorm block
#add_biases_prefixes, \          #True/False, whether to add new biases containing the layernorm's weights
#ln_memory_index, \              #memory index where we store the layernorm activations (for backpropagation) during forward pass

class finalgradient_compute(nn.Module):
    def __init__ (self, \
                  config, \
                  model_config, \
                  block, \
                  add_biases_prefixes , \
                  ln_memory_index, \
                 ):
        
        super(finalgradient_compute, self).__init__()
        self.config=config
        self.add_biases_prefixes=add_biases_prefixes
        self.ln_memory_index = ln_memory_index
        self.add_biases_prefixes = add_biases_prefixes
        
        
        if  add_biases_prefixes: self.trainable_biases = []        
        self.reqd_biases = []
        self.trainable_biases = []
        ############## -------------------------- Attention module -------------------------- ##############
        #Modules for forward through attention module
        ln_f = LayerNormForward(config, \
                                din=model_config.hidden_size, \
                                use_softmax=False, \
                                memory_index=self.ln_memory_index, \
                               )
        self.add_module('ln_forward', ln_f)
        
        if  add_biases_prefixes: 
            self.add_biases(block)
        
        self.forward_modules = [ln_f]
        
        #config, din, use_softmax, debug_zero=False, retain_nablay=False, projection_matrix=None, memory_index=-1
        ln_f_backward = LayerNormDescent_Backward(config, \
                                                  din=model_config.hidden_size, \
                                                  use_softmax=False, \
                                                  memory_index=self.ln_memory_index, \
                                                  debug_zero=False, \
                                                 )
        self.add_module('ln_back', ln_f_backward)
        #ln_f_descent  = LayerNormDescent(config, \
        #                                 model_config.hidden_size, \
        #                                 use_softmax=False, \
        #                                 memory_index=self.ln_memory_index, \
        #                                 debug_zero=True
        #                                )
        #self.add_module('ln_descent', ln_f_descent)
        self.backward_modules = [ln_f_backward,]

        self.trainable_biases = nn.ParameterList(self.trainable_biases)
        
        
    def add_biases(self, tensor=[]):
        biases = nn.Parameter(torch.zeros(self.config.num_prefixes, self.config.hidden_size))
        w = tensor.weight
        b = tensor.bias
        
        num_wts_per_blank=w.shape[0] // self.config.num_prefixes 
        din = w.shape[0]
      
        with torch.no_grad():
            reshaped_w = torch.diag(w)
            reshaped_w = reshaped_w.reshape(self.config.num_prefixes, num_wts_per_blank * din)
            biases[: , : num_wts_per_blank * din ] = reshaped_w
            biases[: , num_wts_per_blank * din : num_wts_per_blank * din + din ] = b        
        self.trainable_biases += [biases]
                
    def forward(self, \
                hidden_states, \
                position_embeddings, \
                past_memory_prefixes=None, \
                initial_forward=True, \
               ):
        
        counter = 0
        
        
        memory_prefixes = []
        memory_activations = []
        out_hidden_states = hidden_states
        
        for module in self.forward_modules:
            
            hidden_states = out_hidden_states 
            
           
            if initial_forward and len(self.trainable_biases) > 0:
                #hidden_states[:, :self.config.num_prefixes] += ( self.trainable_biases[counter] - hidden_states[:, :self.config.num_prefixes] )
                hidden_states = torch.cat( [ self.trainable_biases[counter].unsqueeze(0).expand(hidden_states[:, :self.config.num_prefixes].shape), hidden_states[:, self.config.num_prefixes:]  ], axis=1 )
                #hidden_states = torch.cat( [ self.trainable_biases[counter], hidden_states[:, self.config.num_prefixes:]  ], axis=1 )
            else:
                hidden_states = torch.cat( [ past_memory_prefixes[counter], hidden_states[:, self.config.num_prefixes:]  ], axis=1 )
                #hidden_states[:, :self.config.num_prefixes] += ( past_memory_prefixes[counter] - hidden_states[:, :self.config.num_prefixes] )
                
            counter += 1
            #store the weights for the future backward passes
            memory_prefixes += [hidden_states[:, :self.config.num_prefixes]]
            
            out_hidden_states = module (hidden_states, position_embeddings) 
            #store the memory for the future backward passes
            mem_loc = self.ln_memory_index
            memory_activations += [ out_hidden_states[:, self.config.num_prefixes:, mem_loc:] ]
            
        return out_hidden_states, memory_prefixes, memory_activations
    
    def lossgradient(self, \
                     hidden_states, \
                     desd_output,\
                    ):    
        
        #The true gradient should be E(p-q), where E is the embedding matrix, p is the predicted probability 
        #q is the true probability
        #However, this is expensive because it involves softmax, I simply use -Eq.
        

        din = desd_output.shape[-1]
        num_prefixes=self.config.num_prefixes
        #hidden_states[:, num_prefixes:, :din] -= hidden_states[:, num_prefixes:, :din] 
        hidden_states[:, num_prefixes:, :din] -= ( desd_output + hidden_states[:, num_prefixes:, :din]   )
        
        #print (torch.max(desd_output).item() )
        #print (hidden_states[:, 192, :din])
        #exit(0)

        return hidden_states

    
    def backward(self, \
                 hidden_states, \
                 position_embeddings, \
                 attention_mask, \
                 memory_prefixes, \
                 memory_activations, \
                ):    
        
        memory_counter = 0
        counter = len(memory_activations)-1
        new_memory_prefixes = []
        
        for module in self.backward_modules:
            #if self.update_memory[memory_counter]:
            #first copy the weights onto the blank tokens
            #hidden_states[:, :self.config.num_prefixes] += ( memory_prefixes[counter] - hidden_states[:, :self.config.num_prefixes] ) <---- in-place operation
            #print (memory_prefixes[counter].shape)
            hidden_states = torch.cat( [ memory_prefixes[counter], hidden_states[:, self.config.num_prefixes:] ], axis=1 )
            #further copy the memory on activations 
            mem_loc = self.ln_memory_index
            hidden_states[:, self.config.num_prefixes:, mem_loc:] += ( memory_activations[counter] - hidden_states[:, self.config.num_prefixes:, mem_loc:] )
            counter -= 1
            
            hidden_states = module (hidden_states, position_embeddings, attention_mask)
            #push the new weights into memory
            #if self.store_memory[memory_counter]: 
            new_memory_prefixes += [hidden_states[:, :self.config.num_prefixes]]
            memory_counter += 1
        
        return hidden_states, new_memory_prefixes
        
        
        
#This module stacks all the necessary modules for all the layers in the auxiliary model
#model_dict represents a dictionary view of the auxiliary model        
class TinT_opt(nn.Module):
    
    def __init__(self, \
                 config, \
                 model_config, \
                 model_dict, \
                 #num_forward_backward_passes=1, \
                 #num_backward_layers=-1, \
                 #reuse_forward_blocks=False, \
                 #reuse_backward_blocks=False, \
                ):
        
        super(TinT_opt, self).__init__()
        self.model_config=model_config
        self.config=config
        
        
        #reuse_forward_blocks=self.config.reuse_forward_blocks
        #reuse_backward_blocks=self.config.reuse_backward_blocks
        
        
        self.wte = nn.Embedding(self.model_config.vocab_size, self.model_config.hidden_size)
        self.wpe = OPTLearnedPositionalEmbedding(self.model_config.max_position_embeddings, self.model_config.hidden_size)
        
        self.ln_memory_index = self.config.hidden_size -  self.model_config.hidden_size * 2
        self.attn_memory_index = self.config.hidden_size - self.model_config.hidden_size * 3
        self.linear_memory_index = self.config.hidden_size - self.model_config.hidden_size 
        self.act_memory_index = self.config.hidden_size - self.model_config.hidden_size 
        
        self.wte.weight = model_dict['model']['decoder']['embed_tokens'].weight
        self.wpe.weight = model_dict['model']['decoder']['embed_positions'].weight

        
        self.lm_head = torch.nn.Linear(in_features=self.model_config.hidden_size, \
                                       out_features=self.model_config.vocab_size, \
                                       bias=False \
                                      )
        self.lm_head.weight = model_dict['lm_head'].weight
        self.drop = nn.Dropout(config.embd_pdrop)
        
        if config.n_debug_layers != -1:
            self.n_layers = config.n_debug_layers
        else:
            self.n_layers = self.model_config.num_hidden_layers
            
        if config.n_simulation_layers != -1:
            self.n_back_layers = config.n_simulation_layers
        else:
            self.n_back_layers = self.model_config.n_layer
            
        self.n_forward_backward = config.n_forward_backward
        
        
        if self.config.device == 'cuda': device='cuda:0'
        else: device = 'cpu'
            
        self.wte.to(device)
        self.wpe.to(device)
        self.lm_head.to(device)

            
        #first set of forward modules
        self.all_modules = []
        self.mlp_inner_projections = []
        self.mlp_outer_projections = []
        for layer in tqdm(range(self.n_layers), desc="Building initial Forward simulation modules"):
            #per_oper_gpu = (self.config.n_gpus // 3)
            
            if self.config.device == 'cuda':
                device = 'cuda:'+str( layer // self.config.n_layers_pergpu )
            else:
                device = 'cpu'
            #print (device, self.n_layers // per_oper_gpu, layer % ( self.n_layers // per_oper_gpu ) )
            module = optblock_forward(config, \
                                       model_config, \
                                       block=model_dict['model']['decoder']['layers'][str(layer)], \
                                       add_biases_prefixes=True, \
                                       ln_memory_index=self.ln_memory_index, \
                                       attn_memory_index=self.attn_memory_index, \
                                       linear_memory_index=self.linear_memory_index, \
                                       act_memory_index=self.act_memory_index, \
                                       layer_id=layer, \
                                      )
            self.add_module('Forwardlayer_'+str(layer+1), module)
            self.all_modules += [module.to(device)]
                                
                
        self.gradient_module = finalgradient_compute(config, \
                                                     model_config, \
                                                     block=model_dict['model']['decoder']['final_layer_norm'], \
                                                     add_biases_prefixes=True, \
                                                     ln_memory_index=self.ln_memory_index, \
                                                    )
        self.add_module('Gradientmodule', self.gradient_module.to(device))
        
        for for_back_iter in range(self.n_forward_backward):
            #first set of backward modules
            #device='cuda:'+str(per_oper_gpu + 1) % (self.config.n_gpus))
            
            
            
            
            for layer in tqdm(range(self.n_layers-self.n_back_layers, self.n_layers), desc="Building Backward simulation modules for iteration "+str(for_back_iter + 1)):
                
                if self.config.device == 'cuda':
                    device = 'cuda:'+str( (self.n_layers + layer - (self.n_layers-self.n_back_layers) ) // self.config.n_layers_pergpu )
                else:
                    device = 'cpu'
                
                
                
                pre_forlayer = for_back_iter * (2 * self.n_back_layers) + layer
                if (not self.config.reuse_backward_blocks) or for_back_iter == 0:
                    module = optblock_backward(config, \
                                                model_config, \
                                                ln_memory_index=self.ln_memory_index, \
                                                attn_memory_index=self.attn_memory_index, \
                                                linear_memory_index=self.linear_memory_index, \
                                                act_memory_index=self.act_memory_index, \
                                                mlp_inner_projection_matrix=self.all_modules[pre_forlayer].inner_projection_matrixes[0],\
                                                mlp_outer_projection_matrix=self.all_modules[pre_forlayer].outer_projection_matrixes[0], \
                                                memory_locations=self.all_modules[pre_forlayer].memory_locations, \
                                               )
                    self.add_module('Backwardlayer_'+str(layer+1)+'_Iter_'+str(1+for_back_iter), module)
                else:
                    module = self.all_modules[ layer + self.n_back_layers ]
                    
                self.all_modules += [module.to(device)]
                
                
                #pre_forlayer = for_back_iter * (2 * self.n_back_layers) + layer
                #module = optblock_backward(config, \
                #                            model_config, \
                #                            ln_memory_index=self.ln_memory_index, \
                #                            attn_memory_index=self.attn_memory_index, \
                #                            linear_memory_index=self.linear_memory_index, \
                #                            act_memory_index=self.act_memory_index, \
                #                            mlp_inner_projection_matrix=self.all_modules[pre_forlayer].inner_projection_matrixes[0], \
                 #                           mlp_outer_projection_matrix=self.all_modules[pre_forlayer].outer_projection_matrixes[0], \
                 #                           memory_locations=self.all_modules[pre_forlayer].memory_locations, \
                 #                          )

                #self.add_module('Backwardlayer_'+str(layer+1)+'_Iter_'+str(1+for_back_iter), module)
                #self.all_modules += [module.to(device)]
                #print (device)

            #final forward pass
            for layer in tqdm(range(self.n_layers-self.n_back_layers, self.n_layers), desc="Building final Forward simulation modules for iteration "+str(for_back_iter + 1)):
                #device='cuda:'+str(2 % (self.config.n_gpus))
                #per_oper_gpu = (self.config.n_gpus // 3)
                #device = 'cuda:'+str( 2*per_oper_gpu + layer // ( self.n_layers // per_oper_gpu ) )
                if self.config.device == 'cuda':
                    device = 'cuda:'+str( (self.n_layers + self.n_back_layers  + layer - (self.n_layers-self.n_back_layers) ) // self.config.n_layers_pergpu )
                else:
                    device = 'cpu'    
            
            
                if not self.config.reuse_forward_blocks:
                    
                    module = optblock_forward(config, \
                                               model_config, \
                                               block=model_dict['model']['decoder']['layers'][str(layer)], \
                                               add_biases_prefixes=False, \
                                               ln_memory_index=self.ln_memory_index, \
                                               attn_memory_index=self.attn_memory_index, \
                                               linear_memory_index=self.linear_memory_index, \
                                               act_memory_index=self.act_memory_index, \
                                               layer_id=layer, \
                                              )
                    self.add_module('Forwardlayer_'+str(layer+1)+'_Iter_'+str(1+for_back_iter), module)
                else:
                    module = self.all_modules[layer]
                self.all_modules += [module.to(device)]
                #print (device)
    
           
    def loss_prediction(self, \
             hidden_state, \
             target,\
            ):
        
        device=hidden_state.device
        loss_fct = torch.nn.CrossEntropyLoss()
        
        prediction = self.lm_head( hidden_state[:, self.config.num_prefixes:, :self.model_config.hidden_size] )
        target = target.to(dtype=torch.long)

        loss = loss_fct( prediction[:, :-1].reshape((-1, self.model_config.vocab_size)), target[:, 1:].reshape((-1,))  )
        return prediction, loss
    
    #bidrectional mask: 1 for places where we allow gradient descent, 0 otherwise
    def forward(self, \
                input_ids, \
                bidirection_mask, \
                gradient_mask=None, \
                icl_mask=None, \
                position_ids=None, \
                continue_from_first_forward_pass=False, \
                test_backward_pass=True, \
                labels=None,\
                pad_token=-1, \
               ):        
        device = input_ids.device 
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        batch_size = input_ids.shape[0]   
        
        
        
        attention_mask = bidirection_mask.view(batch_size, -1)
        batch_seq_length = len(attention_mask[0]) 
        total_seq_length = batch_seq_length + self.config.num_prefixes
        causal_mask = torch.tril( torch.ones( ( total_seq_length, total_seq_length ) ) ).to(attention_mask.device)
        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        
        #allow bidirectional attention on prefixes
        causal_mask[:self.config.num_prefixes, :self.config.num_prefixes] += 1.
        
        attention_mask = attention_mask[:, None, None, :]
        causal_mask = causal_mask[None, None, :, :].repeat( (attention_mask.shape[0], 1, 1, 1) )
        
        causal_mask[:, :, :, self.config.num_prefixes: total_seq_length] += attention_mask
        attention_mask = torch.clip(causal_mask, 0., 1.).to(device) 
        
        
        #position_embed
        position_embeds = self.wpe(torch.ones(batch_size, batch_seq_length, device=attention_mask.device), position_ids, 0)
        
        #Keep position embeddings trainable as well
        position_embeddings = torch.zeros((batch_size, total_seq_length, self.config.position_dim), device=device, requires_grad=False)
        position_embeddings[:, self.config.num_prefixes:, :batch_seq_length] = torch.eye(batch_seq_length)
        position_embeddings[:, :self.config.num_prefixes, self.config.seq_length:] = torch.eye(self.config.num_prefixes)
        
        
        #print ( input_ids.device, next(self.wte.parameters()).device )
        input_embeds = self.wte(input_ids)
        
        
        original_hidden_states = input_embeds + position_embeds
        original_hidden_states = self.drop(original_hidden_states)
        
        #desd_output is target for computing gradient of the loss function!
        desd_output = torch.zeros_like(input_embeds)
        desd_output[:, :-1] += (input_embeds[:, 1:])
        #desd_output = desd_output * bidirection_mask.unsqueeze(dim=-1).to(device) 

        
        hidden_states = torch.zeros((original_hidden_states.shape[0], total_seq_length, self.config.hidden_size)).to(device)
        hidden_states[:, self.config.num_prefixes:, :self.model_config.hidden_size] += original_hidden_states
        
        
        #return hidden_states
        
        #first forward pass
        #print (hidden_states[0, 192])
        memory_blank_stack = []
        memory_activations_stack = []
        for layer in range(self.n_layers):
        #tqdm(range(self.n_layers), desc="Initial forward pass"):
            
            if layer == self.n_layers - self.n_back_layers:
                continued_forward_state=hidden_states
            
            device=next(self.all_modules[layer].parameters()).device
            
            hidden_states, \
            memory_prefixes, \
            memory_activations = self.all_modules[layer].forward(hidden_states=hidden_states.to(device),\
                                                                 position_embeddings=position_embeddings.to(device),\
                                                                 initial_forward=True, \
                                                                 icl_mask=icl_mask,
                                                                )
            #print (hidden_states[0, 192])
            if layer >= self.n_layers - self.n_back_layers:
                memory_activations_stack += [ memory_activations ]
                memory_blank_stack += [ memory_prefixes ]
            else:
                memory_activations_stack += [ None ]
                memory_blank_stack += [ None ]
                
        self.memory_activations_stack = memory_activations_stack
        
        original_loss=None
        
        
        
        if test_backward_pass:
            
            original_forward_state=hidden_states
            if continue_from_first_forward_pass:
                continued_forward_state=hidden_states
            
            


            device=next(self.gradient_module.parameters()).device
            
            hidden_states, \
            gradient_blank, \
            gradient_act = self.gradient_module.forward(hidden_states=hidden_states.to(device), \
                                                        position_embeddings=position_embeddings.to(device) \
                                                       )

            if labels is not None:
                device=next(self.lm_head.parameters()).device
                original_logits, original_loss = self.loss_prediction(hidden_states.to(device), target=labels.to(device))
                    

            
            
            for for_back_iter in range(self.n_forward_backward):
                #compute loss gradient
                #device=next(self.gradient_module.parameters()).device
                
                
                device=next(self.gradient_module.parameters()).device
                if self.config.use_prediction_loss:
                    lm_head_device=next(self.lm_head.parameters()).device
                    prediction = torch.nn.Softmax(dim=-1)( self.lm_head( hidden_states[:, self.config.num_prefixes:, :self.model_config.hidden_size].to(lm_head_device) ) ) @ self.lm_head.weight 
                    gradient = ( desd_output -  prediction.to( desd_output.device ) ).to(device)
                   
                elif self.config.use_quad:
                    quad_pred = self.lm_head( hidden_states[:, self.config.num_prefixes:, :self.model_config.hidden_size].to(device) ) @ self.lm_head.weight
                    gradient = ( desd_output - quad_pred.to( desd_output.device  ) ).to(device)
 
                else:
                    gradient = ( desd_output - torch.mean(self.lm_head.weight, axis=0)[None, None, :].to( desd_output.device ) ).to(device)
                
                #mask out tokens in the sequence that you don't to compute loss on
                gradient[:, :-1] *= bidirection_mask[:, 1:].unsqueeze(dim=-1).to(device) 
                if gradient_mask is not None:
                    gradient[:, :-1] *= gradient_mask[:, 1:].unsqueeze(-1)
                hidden_states = self.gradient_module.lossgradient( hidden_states.to(device), gradient ) 

                
                #device=next(self.gradient_module.parameters()).device 
                #hidden_states = hidden_states.to(device)
                #Backward pass
                hidden_states, \
                lnf_memory_prefixes = self.gradient_module.backward(hidden_states=hidden_states.to(device), \
                                                                  position_embeddings=position_embeddings.to(device), \
                                                                  attention_mask=attention_mask.to(device),\
                                                                  memory_activations=[p.to(device) for p in gradient_act], \
                                                                  memory_prefixes=[p.to(device) for p in gradient_blank], \
                                                                 )
                #print (lnf_memory_prefixes[0][:, 0] - gradient_blank[0][:, 0])
                
                #device='cuda:'+str(1 % self.config.n_gpus) 
                #hidden_states = hidden_states.to(device)
                module_begin = for_back_iter * (2 * self.n_back_layers) + self.n_layers
                for layer in range(self.n_layers-1, self.n_layers-self.n_back_layers-1, -1):
                #tqdm(range(self.n_layers-1, self.n_layers-self.n_back_layers-1, -1), \
                #                  desc="Backward and descent, Iteration " +str(for_back_iter+1)):
                    
                    #stack_layer=2*self.n_layers - 1 - layer
                    
                    module_index = module_begin + layer - (self.n_layers-self.n_back_layers)
                    device=next(self.all_modules[module_index].parameters()).device 
                    
                    #stack_layer = for_back_iter * (2 * self.n_back_layers) + self.n_layers - 1 - layer
                    hidden_states, \
                    memory_prefixes = self.all_modules[module_index].forward(hidden_states=hidden_states.to(device),\
                                                                           position_embeddings=position_embeddings.to(device),\
                                                                           attention_mask=attention_mask.to(device), \
                                                                           memory_activations=[p.to(device) for p in memory_activations_stack[layer]], \
                                                                           memory_prefixes=[p.to(device) for p in memory_blank_stack[layer]], \
                                                                           icl_mask=icl_mask,\
                                                                          )
                    #print (  [torch.amax( torch.abs( memory_blank_stack[layer][i] - memory_prefixes[i] ) )  for i in range(len(memory_prefixes))] )
                    #exit(0)
                    memory_blank_stack[layer] = memory_prefixes

                #Final forward pass
                #if test_entire_model:
                module_begin  = module_begin + self.n_back_layers
                hidden_states = continued_forward_state
                #hidden_states = hidden_states.to( 'cuda:'+str(2 % self.config.n_gpus) )
                
                for layer in range(self.n_layers-self.n_back_layers, self.n_layers):
                    #tqdm(range(self.n_layers-self.n_back_layers, self.n_layers), \
                    #     desc="Forward pass, Iteration " +str(for_back_iter+1)):
                    
                    #stack_layer=layer - 2*self.n_layers
                    #device=next(self.all_modules[module_index].parameters()).device 
                    
                    module_index = module_begin + layer - (self.n_layers-self.n_back_layers)
                    device=next(self.all_modules[module_index].parameters()).device 
                    
                    hidden_states, \
                    _, \
                    memory_activations = self.all_modules[module_index].forward(hidden_states=hidden_states.to(device),\
                                                                                position_embeddings=position_embeddings.to(device),\
                                                                                past_memory_prefixes=[p.to(device) for p in memory_blank_stack[layer]],\
                                                                                initial_forward=False, \
                                                                                icl_mask=icl_mask,\
                                                                               )
                    #memory_blank_stack[layer]       = memory_prefixes
                    memory_activations_stack[layer] = memory_activations
                    
                #print ( max([ torch.max( gradient_blank[0][:, i] - lnf_memory_prefixes[0][:, i] ).item() for i in range(self.config.num_prefixes) ]))  
                #exit(0)
                
                #print (hidden_states[0, 192])
                #print ('******', len(lnf_memory_prefixes), '******')
                device=next(self.gradient_module.parameters()).device 
                hidden_states, \
                gradient_blank, \
                gradient_act = self.gradient_module.forward(hidden_states=hidden_states.to(device), \
                                                            position_embeddings=position_embeddings.to(device),\
                                                            past_memory_prefixes=[p.to(device) for p in lnf_memory_prefixes],\
                                                            initial_forward=False, \
                                                           )
            
                
                
        final_loss = None    
        if labels is not None:
            device=next(self.lm_head.parameters()).device 
            final_logits, final_loss = self.loss_prediction(hidden_states.to(device), target=labels.to(device))
        
        from argparse import Namespace
        return Namespace( original_loss=original_loss, final_loss=final_loss, original_logits=original_logits, logits=final_logits)
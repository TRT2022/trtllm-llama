'''

对比hf模型和ft模型权重是否一致

'''

import numpy as np


# #not same
# tf = np.load("./hf/model.layers.0.mlp.gate_proj.weight.npy")
# print("-------------")
# print(tf)
# print(tf.shape)

#not same
hf = np.load("./hf/model.layers.1.self_attn.qkv_proj.weight.npy")
tf = np.load("./ft/model.model.layers.1.attention.query_key_value.weight.bin.npy").transpose([1,0])

print(hf)
print(hf.shape)
print("-------------")
print(tf)
print(tf.shape)
print(tf==hf)
print(np.sum(tf!=hf))

# not same
hf = np.load("./hf/model.layers.8.mlp.gate_proj.weight.npy")
tf = np.load("./ft/model.model.layers.8.mlp.gate_proj.weight.0.bin.npy").transpose([1,0])

print(hf)
print(hf.shape)
print("-------------")
print(tf)
print(tf.shape)

print(tf==hf)
print(np.sum(tf!=hf))

<----------------------------------------------------------------------------

# same
hf = np.load("./hf/model.layers.0.self_attn.o_proj.weight.npy")
tf = np.load("./ft/model.model.layers.0.attention.dense.weight.0.bin.npy").transpose([1,0])

print(hf)
print(hf.shape)
print("-------------")
print(tf)
print(tf.shape)
print(tf==hf)
print(np.sum(tf!=hf))

# same
hf = np.load("./hf/model.layers.0.input_layernorm.weight.npy")
tf = np.load("./ft/model.model.layers.0.input_layernorm.weight.bin.npy")

print(hf)
print(hf.shape)
print("-------------")
print(tf)
print(tf.shape)

# same
hf = np.load("./hf/model.layers.0.mlp.down_proj.weight.npy")
tf = np.load("./ft/model.model.layers.0.mlp.down_proj.weight.0.bin.npy").transpose([1,0])

print(hf)
print(hf.shape)
print("-------------")
print(tf)
print(tf.shape)

print(tf==hf)


# same
hf = np.load("./hf/model.layers.8.mlp.up_proj.weight.npy")
tf = np.load("./ft/model.model.layers.8.mlp.up_proj.weight.0.bin.npy").transpose([1,0])

print(hf)
print(hf.shape)
print("-------------")
print(tf)
print(tf.shape)

print(tf==hf)

# same
hf = np.load("./hf/model.layers.8.post_attention_layernorm.weight.npy")
tf = np.load("./ft/model.model.layers.8.post_attention_layernorm.weight.bin.npy")

print(hf)
print(hf.shape)
print("-------------")
print(tf)
print(tf.shape)

print(tf==hf)


# same
hf = np.load("./hf/model.norm.weight.npy")
tf = np.load("./ft/model.final_layernorm.weight.bin.npy")

print(hf)
print(hf.shape)
print("-------------")
print(tf)
print(tf.shape)

print(tf==hf)



# same
hf = np.load("./hf/lm_head.weight.npy")
tf = np.load("./ft/model.lm_head.weight.bin.npy")

print(hf)
print(hf.shape)
print("-------------")
print(tf)
print(tf.shape)

print(tf==hf)

# same
hf = np.load("./hf/model.embed_tokens.weight.npy")
tf = np.load("./ft/model.wte.weight.bin.npy")

print(hf)
print(hf.shape)
print("-------------")
print(tf)
print(tf.shape)

print(tf==hf)
print(np.sum(tf!=hf))



# C++ GPT Runtime

TensorRT-LLM includes a C++ component to execute TensorRT engines built with
the Python API as described in the [Architecture](architecture.md) document.
That component is called the C++ runtime.

_In this release of TensorRT-LLM, the C++ and Python runtimes are two separate
software components. The Python runtime will be refactored and built on top of
the C++ runtime in a future release_.

The API of the C++ runtime is composed of the classes declared in
[`cpp/include/tensorrt_llm/runtime`](../cpp/include/tensorrt_llm/runtime) and
implemented in
[`cpp/tensorrt_llm/runtime`](../cpp/tensorrt_llm/runtime). An example of
how to use the C++ runtime for a GPT-like auto-regressive model can be found in
[`cpp/tests/runtime/gptSessionTest.cpp`](../cpp/tests/runtime/gptSessionTest.cpp).

## The Session

The main component of the C++ runtime is the session. For GPT-like
auto-regressive models, it is implemented by the
[`GptSession`](../cpp/include/tensorrt_llm/runtime/gptSession.h) class.

### Creation

The constructor of that class allows users to specify the model and the
environment to execute it. The model is described by an instance of the
[`GptModelConfig`](../cpp/include/tensorrt_llm/runtime/gptModelConfig.h)
class and a pointer to the TensorRT engine that must be
executed to perform the inference. The environment is configured through the
[`WorldConfig`](../cpp/include/tensorrt_llm/runtime/worldConfig.h)
(that name comes from
[MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface) and its "famous"
`MPI_COMM_WORLD` default communicator). The constructor also accepts an
optional object to log informations, warnings and errors:

```cpp
#include <tensorrt_llm/runtime/gptSession.h>

using namespace tensorrt_llm::runtime;

GptSession session(modelConfig,   // Description of the model,
                   worldConfig,   // Description of the environment,
                   engineBuffer,  // The compiled TensorRT engine (const void*),
                   engineSize,    // The size in bytes of the TensorRT engine (size_t),
                   logger);       // The optional logger.
```

The above constructor accepts a `const void*` pointer to the engine and the
associated size (in bytes) of that buffer. There exist other overloaded
versions that take `std::vector<uint8_t>` or `std::string` arguments to
encapsulate the engine.

#### Model Configuration

The model configuration is an instance of the
[`GptModelConfig`](../cpp/include/tensorrt_llm/runtime/gptModelConfig.h)
class. In this release, that class encapsulates the following parameters:

 * `vocabSize`, the size of the vocabulary,
 * `numLayers`, the number of layers in the model,
 * `numHeads`, the number of heads in the attention block,
 * `hiddenSize`, the size of the hidden dimension,
 * `dtype`, the data-type that was used to build the TensorRT engine and that
   must be used to run the model during inference,
 * `useGptAttentionPlugin`, indicates if the [GPT Attention](gpt_attention.md)
   operator was compiled using the
   [GPT Attention plugin](../cpp/tensorrt_llm/plugins/gptAttentionPlugin),
 * `useMultiQueryMode`, indicates if the model uses multi-query attention (MQA)
   instead of multihead attention (MHA). See [GPT Attention](gpt_attention.md),
 * `inputPacked`, indicates that the input must be packed (or padded when set
   to `false`). For performance reasons, it is recommended to always use packed,
   even if its default is set to `false` (will be changed in a future release).
   See [GPT Attention](gpt_attention.md),
 * `useInflightBatchingGptAttentionPlugin`, indicates that the TensorRT-LLM
   model was built using the special plugin for in-flight batching. That flag
   will be removed in the next release of TensorRT-LLM when the "regular" GPT
   Attention plugin and the custom in-flight batching plugin are merged. See
   [GPT Attention](gpt_attention.md) and
   [In-flight Batching](in_flight_batching.md) for details.

Please note that, as mentioned in the above description, some of the fields in
that class will be removed in the future release of TensorRT-LLM and some
default values will be changed.

#### World Configuration

Familiarity with
[MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface), is not required
to utilize the TensorRT-LMM C++ runtime. There are two main things
you need to know: (1) The C++ Runtime in TensorRT-LLM uses
[processes](https://en.wikipedia.org/wiki/Process_(computing)) to execute
TensorRT engines on the different GPUs. Those GPUs can be located on a single
node as well as on different nodes in a cluster. Each process is called a
*rank* in MPI. (2) The ranks are grouped in communication groups. The
TensorRT-LLM C++ Runtime calls that group the *world*.

The world configuration is an instance of the
[`WorldConfig`](../cpp/include/tensorrt_llm/runtime/worldConfig.h)
class. In this release, that class encapsulates the following parameters:

* `worldSize`, is the number of ranks that collaborate together,
* `rank`, is the unique identifier of the rank (see below),
* `gpusPerNode`, indicates the number of GPUs on each node. Having that
  information allows the C++ runtime to optimize communications between GPUs in
  a node (like taking advantage of the
  [NVLINK](https://www.nvidia.com/en-us/data-center/nvlink/)
  interconnect between GPUs of an A100
  [DGX](https://www.nvidia.com/en-us/data-center/dgx-platform/)
  node).

For a multi-GPU configuration (single or multi-node), each rank must create its
own instance of `GptSession` using its own `WorldConfig`. A typical example
is:

```cpp
#include <mpi.h>

// Initialize the MPI library.
MPI_Init(&argc, &argv);

// Get the number of ranks (size of the world).
int worldSize;
MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

// Get the unique identifier for each rank.
int rank;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);

// Create the TensorRT-LLM Runtime WorldConfig.
tensorrt_llm::runtime::WorldConfig worldConfig(worldSize, rank);

// Create the GPT session (as shown above).
tensorrt_llm::runtime::GptSession session(modelConfig, worldConfig, ...);
```

Once compiled, that C++ code must be executed using the `mpirun` command
installed on the system (talk to your system administrator if needed):

```bash
# Launch the program using two processes (worldSize == 2 and ranks == {0, 1}).
mpirun -n 2 ...
```

### Setup

The `GptSession::setup` member function must be called to prepare the runtime
to execute the inference on a batch of input sequences. That member function
takes four arguments:

 * `batchSize`, the number of sequences in the batch,
 * `maxInputLength`, the length of the longest input sequence,
 * `maxNewTokens`, an upper bound on the number of tokens that can be generated,
 * `samplingConfig`, a structure that contains sampling parameters (see below).

The [`SamplingConfig`](../cpp/include/tensorrt_llm/runtime/samplingConfig.h)
class encapsulates parameters that control the
[generation](https://huggingface.co/blog/how-to-generate) of new tokens.
Except for the `beamWidth` parameter, all the fields are optional and the
runtime will use a default value if no values are provided by the user. For
vector fields, the TensorRT-LLM runtime supports one value per sequence (i.e.
the vector contains `batchSize` values). If all the sequences use the same
value for a given parameter, the vector can be limited to a single element
(i.e. `size() == 1`).

***General***

 * `temperature`, a vector of floating-point numbers to control the
   modulation of logits when sampling new tokens. The default value is `1.0f`,
 * `minLength`, a vector of integers to set a lower-bound on the number of tokens
   generated. The default value is 1,
 * `repetitionPenalty`, a vector of float-point numbers to penalize tokens
   based on how often they appear in the sequence. The default value is `0.f`,
 * `presencePenalty`, a vector of float-point numbers to penalize tokens
   already present in the sequence (irrespective of the number of appearances).
   The default value is `0.f`,

The parameters `repetitionPenalty` and `presencePenalty` are mutually
exclusive. In this release, it means that a user can only set, at most, one of
those two optional fields.  In a future release, we might adopt a finer-grained
method based on checking the values.

***Sampling***

 * `randomSeed`, a vector of 64-bit integers to control the random seed used by
   the random number generator in sampling. Its default value is 0,
 * `topK`, a vector of integers to control the number of logits to sample from.
   Its default value is 0. Note that if different values are provided for the
   different sequences in the batch, the performance of the implementation will
   depend on the largest value. For efficiency reasons, we recommend to batch
   requests with similar `topK` values together,
 * `topP`, a vector of floating-point values to control the top-P probability
   to sample from. Its default value is `0.f`,
 * `topPDecay`, `topPMin` and `topPResetIds`, vectors to control the decay in
   the top-P algorithm. The top-P values are modulated by
   a decay that exponentially depends on the length of the sequence as explained in
   [_Factuality Enhanced Language Models for Open-Ended Text Generation_](https://arxiv.org/abs/2206.04624).
   `topPDecay` is the decay, `topPMin` is the lower-bound and `topPResetIds`
   indicates where to reset the decay. Defaults are `1.f`, `1.0e-6,f` and `-1`,

If both `topK` and `topP` fields are set, the top-K method will be run for
sequences with a `topK` value greater than `0.f`. In that case, the `topP`
value for that sequence also influences the result. If the `topK` values for
some sequences are `0.f`, the top-P method will be used for those remaining
sequences. If both `topK` and `topP` are zero, greedy search is performed.

***Beam-search***

 * `beamWidth`, is the width used for the [beam
   search](https://en.wikipedia.org/wiki/Beam_search) sampling algorithm. There
   is no explicit upper-bound on the beam width but increasing the beam width
   will likely increase the latency. Use 1 to disable beam-search,
 * `beamSearchDiversityRate`, a floating-point value that controls the
   diversity in beam-search. Its default value is `0.f`,
 * `lengthPenalty`, a floating-point value that controls how to penalize the
   longer sequences in beam-search (the log-probability of a sequence will be
   penalized by a factor that depends on `1.f / (length ^ lengthPenalty)`). The
   default is value `0.f`. The parameter `lengthPenalty` may be renamed to
   `beamSearchLengthPenalty` in a future release,

The `beamWidth` parameter is a scalar value. It means that in this release of
TensorRT-LLM, it is not possible to specify a different width for each input
sequence. This limitation is likely to be removed in a future release.

### Generation

The `GptSession::generate` member function performs the generation loop. Given
input tensors to read from, output tensors to populate, that member function
will run the generation loop until it reaches the maximum number of tokens that
can be produced or each sequence has reached completion (due to the production
of "end-of-sequence" or a word in the list of "stop words"). The pseudo-code of
that function looks like (member function names were changed to keep the
presentation simple):

```cpp
// Have all the sequences in the batch reached completion?
bool allFinished = false;

// Until all sequences are finished or the number of steps reaches the limit...
for (int step = 0; !allFinished && step < maxNewTokens; ++step) {

  // Trigger the computation of the logits...
  computeLogits(...);

  // Run the sampling to produce a token (for each active sequence) from the logits.
  allFinished = generateTokensFromLogits(...);

  // Callback to stream the output tokens while the generation loop continues.
  onTokenGenerated(...);
}
```

#### Inputs and Outputs

The `generate` member function takes an instance of the
[`GenerationInput`](../cpp/include/tensorrt_llm/generationInput.h) class and
populates an instance of the
[`GenerationOutput`](../cpp/include/tensorrt_llm/generationOutput.h) class.

***Mandatory inputs***

 * `endId`, is the token ID that marks the end of the input sequence (aka `EOS`
   or end-of-sequence). It's `50,256` for the GPT2 model which has a vocabulary
   of `50,257` tokens, for example,
 * `padId`, is the token ID that is used for padding (i.e. fills in the slots
   that are at an index greater-or-equal to the input length for padded
   sequences). It can be set to the same value as `endId`,
 * `ids`, is the tensor of input IDs. That tensor must be allocated on the GPU.
   When the input tensor is padded, the shape of `ids` is `[batchSize,
   maxInputLength]`, where `batchSize` and `maxInputLength` correspond to the
   arguments passed to the `GptSession::setup` member function. When the input
   is packed, the shape of `ids` is `[numTokens]`, where `numTokens` is the sum
   of the lengths of the different sequences in the batch,
 * `lengths`, is the tensor of input sequence lengths. That tensor must be
   allocated on the GPU and contain `batchSize` values,
 * `packed`, indicates if the `ids` tensor is packed or padded. In this
   release, that flag must match the value passed to the constructor through
   the instance of the `ModelConfig` class. In a future release, the session
   may be made more flexible and automatically pad or pack the input,

***Optional inputs***

 * `embeddingBiasOpt`, is a tensor of floating-point values on the GPU that
   contains the bias to add to the logits during sampling (after the projection
   from hidden states to logits as the last step of the model). This tensor
   must have `vocabSize` elements (as defined in the `ModelConfig` argument
   passed to the constructor),
 * `badWordsList`, is a tensor of integers on the GPU that encodes the list of
   words that have to be banned from generated sequences. Its shape is `[2,
   badWordsLength]`, as explained below, or `[batchSize, 2, badWordsLength]`
   when there is a different list for each sequence in the batch,
 * `stopWordsList`, is a tensor of integers on the GPU that encodes the list of
   words that trigger the end of the generation for a sequence. Its shape is
   `[2, stopWordsLength]`, as explained below, or `[batchSize, 2,
   stopWordsLength]` when there is a different list for each sequence in the
   batch,

The `badWordsList` and `stopWordsList` tensors have the same shape `[2,
length]`. Let's consider an example with three words to describe the
representation of those lists.  The first word contains tokens `[5, 7, 3]`, the
second one contains `[9, 2]` and the third one is composed of tokens `[6, 2, 4,
1]`. In total, there are 9 tokens. That's the length. The shape of the tensor
is `[2, 9]`.  The first row of the tensor must contain the 9 token IDs and the
second row must store the
[exclusive prefix-sum](https://en.wikipedia.org/wiki/Prefix_sum)
of the word lengths as shown on the following diagram:

```
   0           3       5              9
   |           |       |              |
   V           V       V              V
[  5,  7,  3,  9,  2,  6,  2,  4,  1]
[  0,  3,  5,  9, -1, -1, -1, -1, -1]
```

In case all the words are made of a single token, the inner-most dimension of
the tensor must be increased by 1 (i.e. the length for 4 words, each made of a
single token, must be 5 instead of 4 -- the shape is `[2, 5]`).

***Mandatory outputs***

 * `ids`, is a tensor that contains the output token IDs. Its shape is
   `[batchSize, beamWidth, maxSeqLength]` where `maxSeqLength` is the sum of
   `maxInputLength` and `maxNewTokens`. After generation, it contains, for each
   sequence, a copy of the input tokens followed by the output tokens. When a
   sequence is shorter than `maxSeqLength`, padding tokens are added at the end
   of the sequence.

_Note that the shape of that tensor is different in this version of
TensorRT-LLM from its shape in previous versions where it was `[maxSeqLength,
batchSize, beamWidth]`_.

***Optional outputs***

 * `logProbs`, is a tensor of floating-point values on the GPU to store the
   log-prob of the generated tokens. Its shape is `[maxNewTokens, batchSize,
   beamWidth]`. Its shape will likely change in a future release to match the
   shape of the output `ids` tensor,
 * `onTokenGenerated`, is a callback function invoked in the generation loop to
   pass newly generated tokens to the caller while the loop continues to
   execute. An implementation of that callback must accept the output `ids`
   tensor, the generation `step` and a boolean flag that indicates if the
   generation is complete.

## Internal Components

The `GptSession` class encapsulates two main components. The
[`TllmRuntime`](../cpp/tensorrt_llm/runtime/tllmRuntime.h) is in charge of the
execution of the TensorRT engine. The
[`GptDecoder`](../cpp/include/tensorrt_llm/runtime/gptDecoder.h)
does the generation of the tokens from the logits.  The `TllmRuntime` class is
an internal component and users are not expected to use that class directly.
The `GptDecoder` can be used directly to implement very custom generation loop
and for use cases that cannot be satisfied by the implementation in
`GptSession`.

## In-flight Batching Support

In this release, in-flight batching is supported using a separate session class
`GptSessionWithDecoderBatch`. The biggest difference with the `GptSession`
class is in how it manages the token generation from logits. To be able to add
support for in-flight batching on time for that release, we have modified the
decoder to treat each request separately (i.e. the batch is split into
`batchSize` individual requests) and issue kernels using separated CUDA streams.
This behavior will be revisited in a future release to maintain the structure
of the batch and improve efficiency.

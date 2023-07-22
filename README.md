# llama.cpp-bindings

llama.cpp bindings for Haskell.

## Examples

See examples/Main.hs -- attempts to mimic a subset of the functionality of llama.cpp's main example functionality:

```bash
> result/bin/examples +RTS -xc -RTS -m ../models/llama-2-7b.ggmlv3.q5_K_M.bin --n_ctx 2048 --temp 0.7 -t 8 --n_gpu_layers 32 -p "### Instruction: How awesome is Haskell?\n### Response:"

initBackend

init context params

loading model
llama.cpp: loading model from ../models/llama-2-7b.ggmlv3.q5_K_M.bin
llama_model_load_internal: format     = ggjt v3 (latest)
llama_model_load_internal: n_vocab    = 32000
llama_model_load_internal: n_ctx      = 2048
llama_model_load_internal: n_embd     = 4096
llama_model_load_internal: n_mult     = 256
llama_model_load_internal: n_head     = 32
llama_model_load_internal: n_layer    = 32
llama_model_load_internal: n_rot      = 128
llama_model_load_internal: freq_base  = 10000.0
llama_model_load_internal: freq_scale = 1
llama_model_load_internal: ftype      = 17 (mostly Q5_K - Medium)
llama_model_load_internal: n_ff       = 11008
llama_model_load_internal: model size = 7B
llama_model_load_internal: ggml ctx size =    0.08 MB
llama_model_load_internal: mem required  = 6232.95 MB (+ 1026.00 MB per state)

loading context
llama_new_context_with_model: kv self size  = 2048.00 MB

System Info:
AVX = 1 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | FMA = 1 | NEON = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 1 | VSX = 0 | 

Prompt
### Instruction: How awesome is Haskell?\n### Response:



Tokenized 18

Running first eval of entire prompt

sampling
 Really fucking cool!

### Instruction: What do you think of [this book](http://www.amazon.com/Learn-You-a-Haskell-Weekends/dp/098

freeing context, model, context params

llama_print_timings:        load time =  3510.70 ms
llama_print_timings:      sample time =    22.63 ms /    50 runs   (    0.45 ms per token,  2209.26 tokens per second)
llama_print_timings: prompt eval time =  1879.72 ms /    18 tokens (  104.43 ms per token,     9.58 tokens per second)
llama_print_timings:        eval time =  6072.97 ms /    50 runs   (  121.46 ms per token,     8.23 tokens per second)
llama_print_timings:       total time =  8298.26 ms
$ 
```

## Status

Right now in a super alpha state, here's a summary:
* llama.h API (as of https://github.com/ggerganov/llama.cpp/commit/e782c9e735f93ab4767ffc37462c523b73a17ddc) is largely ported, and insofar as the simple example code exercises it, it works. Speaking of...
* examples/Main.hs implements a subset of llama.cpp's [main example](https://github.com/ggerganov/llama.cpp/blob/e782c9e735f93ab4767ffc37462c523b73a17ddc/examples/main/main.cpp) (a.k.a. the main llama build target) in Haskell. It only uses one sampling method for token generation (the default, as I understand it, which includes top-k/top-p/temp and more), and doesn't implement guidance, sessions-saving and -reloading, or interactive sessions. Yet.  
* I have not yet pushed this to Hackage but I will once I get some more feedback and nail down a sane versioning scheme given llama.cpp's...aggressive pace of development.


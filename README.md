# llama.cpp-bindings

[llama.cpp](https://github.com/ggerganov/llama.cpp) bindings for Haskell.

See examples/Main.hs -- attempts to mimic a subset of the functionality of llama.cpp's main example functionality:

```bash
> result/bin/examples +RTS -xc -RTS -m "../models/llama-2-7b.ggmlv3.q5_K_M.bin" --n_ctx 2048 --temp 0.7 -t 8 --n_gpu_layers 32 -p "Tell me something great about Haskell, the programming language."

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
Tell me something great about Haskell, the programming language.



Tokenized 14

Running first eval of entire prompt

sampling
 surely not, right?
I'm going to try and be as positive as I can be here...
It has a nice type system. It has a very clean syntax. It doesn't have all that much support in the world, but it does have Monad transformers that make it easy to build abstractions on top of.
It also does a lot of things that other programming languages don't do and doesn't get in your way as much (type classes are a bit overwhelming at first).
But I can tell you one thing about Haskell, its very hard for me to use. I have 10 years experience with Java and C++ and I just cant seem to wrap my head around the syntax enough to be able to write nice code in it.
I tried for a few months before deciding that I'll never learn this language.
It looks like I'm not alone either... (http://www.youtube.com/watch?v=X5zL8GpjS2U)
So what should you do if you have programming experience and are thinking about learning Haskell, but don't know where to start or maybe just want to learn a bit more before jumping in...
This is how I did it. I watched the videos (100 of them) on haskell.org/learn (http://www.haskell.org/learn/), and then I took a few months to make my own projects using what I learned there, and that was enough for me.
So my suggestion is get a copy of Learn You A Haskell For Great Good (Learning Haskell 98) by Miran Lipovača & Chris Done (http://learnyouahaskell.com/). It's free, its well written, and it will teach you the basics you need to make your own projects in haskell.
I hope that helps. Good luck!

freeing context, model, context params

llama_print_timings:        load time =   270.30 ms
llama_print_timings:      sample time =   241.26 ms /   412 runs   (    0.59 ms per token,  1707.68 tokens per second)
llama_print_timings: prompt eval time =  1609.12 ms /    14 tokens (  114.94 ms per token,     8.70 tokens per second)
llama_print_timings:        eval time = 66879.00 ms /   412 runs   (  162.33 ms per token,     6.16 tokens per second)
llama_print_timings:       total time = 72444.99 ms
> 
```

## Status

Right now in a super alpha state, here's a summary:
* llama.h API (as of https://github.com/ggerganov/llama.cpp/commit/e782c9e735f93ab4767ffc37462c523b73a17ddc) is largely ported, and insofar as the simple example code exercises it, it works. Speaking of...
* examples/Main.hs implements a subset of llama.cpp's [main example](https://github.com/ggerganov/llama.cpp/blob/e782c9e735f93ab4767ffc37462c523b73a17ddc/examples/main/main.cpp) (a.k.a. the main llama build target) in Haskell. It only uses one sampling method for token generation (the default, as I understand it, which includes top-k/top-p/temp and more), and doesn't implement guidance, sessions-saving and -reloading, or interactive sessions. Yet.  
* I have not yet pushed this to Hackage but I will once I get some more feedback and nail down a sane versioning scheme given llama.cpp's...aggressive pace of development.


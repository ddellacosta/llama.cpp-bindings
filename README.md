# llama.cpp-bindings

llama.cpp bindings for Haskell.

Right now in a super alpha state, here's a summary:
* llama.h API (as of https://github.com/ggerganov/llama.cpp/commit/e782c9e735f93ab4767ffc37462c523b73a17ddc) is largely ported, and insofar as the simple example code exercises it, it works. Speaking of...
* examples/Main.hs implements a subset of llama.cpp's [main example](https://github.com/ggerganov/llama.cpp/blob/e782c9e735f93ab4767ffc37462c523b73a17ddc/examples/main/main.cpp) (a.k.a. the main llama build target) in Haskell. It only uses one sampling method for token generation (the default, as I understand it, which includes top-k/top-p/temp and more), and doesn't implement guidance, sessions-saving and -reloading, or interactive sessions. Yet.  
* I have not yet pushed this to Hackage but I will once I nail down a sane versioning scheme given llama.cpp's...aggressive pace of development.


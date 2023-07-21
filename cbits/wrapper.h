
#include <llama.h>

struct llama_model * wrapper_load_model_from_file(
  const char * path_model,
  struct llama_context_params * params
  );

void wrapper_context_default_params(struct llama_context_params *);

void wrapper_model_quantize_default_params(struct llama_model_quantize_params *);

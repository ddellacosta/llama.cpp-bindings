
#include <llama.h>

struct llama_model * wrapper_load_model_from_file(
  const char * path_model,
  struct llama_model_params * params
  );


struct llama_context * wrapper_new_context_with_model(
            struct llama_model * model,
            struct llama_context_params * params
        );

void wrapper_context_default_params(struct llama_context_params *);

void wrapper_model_default_params(struct llama_model_params *);

void wrapper_model_quantize_default_params(struct llama_model_quantize_params *);

void wrapper_get_timings(struct llama_context * ctx, struct llama_timings * timings);

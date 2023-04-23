import openai

class GPT3:
    def __init__(self, api_key):
        self.LLM_CACHE = {}
        openai.api_key = api_key

    def gpt3_call(self, engine="text-ada-001", prompt="", max_tokens=128, temperature=0, 
                logprobs=1, echo=False):
        full_query = ""
        for p in prompt:
            full_query += p
        id = tuple((engine, full_query, max_tokens, temperature, logprobs, echo))
        if id in self.LLM_CACHE.keys():
            print('cache hit, returning')
            response = self.LLM_CACHE[id]
        else:
            response = openai.Completion.create(engine=engine, 
                                                prompt=prompt, 
                                                max_tokens=max_tokens, 
                                                temperature=temperature,
                                                logprobs=logprobs,
                                                echo=echo)
            self.LLM_CACHE[id] = response
        return response

    def gpt3_scoring(self, query, options, engine="text-ada-001", limit_num_options=None, option_start="\n", verbose=False, print_tokens=False):
        if limit_num_options:
            options = options[:limit_num_options]
        verbose and print("Scoring", len(options), "options")
        gpt3_prompt_options = [query + option for option in options]
        response = self.gpt3_call(
            engine=engine, 
            prompt=gpt3_prompt_options, 
            max_tokens=0,
            logprobs=1, 
            temperature=0,
            echo=True,)
  
        scores = {}
        for option, choice in zip(options, response["choices"]):
            tokens = choice["logprobs"]["tokens"]
            token_logprobs = choice["logprobs"]["token_logprobs"]

            total_logprob = 0
            for token, token_logprob in zip(reversed(tokens), reversed(token_logprobs)):
                print_tokens and print(token, token_logprob)
                if option_start is None and not token in option:
                    break
                if token == option_start:
                    break
            total_logprob += token_logprob
            scores[option] = total_logprob

        for i, option in enumerate(sorted(scores.items(), key=lambda x : -x[1])):
            verbose and print(option[1], "\t", option[0])
            if i >= 10:
                break

        return scores, response

    def make_options(self, pick_targets, place_targets, options_in_api_form=True, termination_string="done()"):
        options = []
        for pick in pick_targets:
            for place in place_targets:
                if options_in_api_form:
                    option = "robot.pick_and_place({}, {})".format(pick, place)
                else:
                    option = "Pick the {} and place it on the {}.".format(pick, place)
            options.append(option)

        options.append(termination_string)
        print("Considering", len(options), "options")
        return options
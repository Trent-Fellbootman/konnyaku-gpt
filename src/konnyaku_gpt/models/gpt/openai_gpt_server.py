import openai

from typing import Sequence, Tuple, Dict
import logging


class OpenAiGptServer:
    
    # {model name: (input cost per 1k tokens, output cost per 1k tokens)}
    price_info = {
        'gpt-3.5-turbo': (0.0015, 0.002),
        'gpt-3.5-turbo-16k': (0.003, 0.004),
        'gpt-4': (0.03, 0.06),
        'gpt-4-32k': (0.06, 0.12),
    }
    
    def __init__(self):
        # {model name: input tokens used, output tokens used}
        self.token_usages = {name: (0, 0) for name in self.price_info.keys()}
    
    
    def invoke(self, model_name: str, messages: Sequence[Tuple[str, bool]]) -> str:
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[
                {"role": "user" if is_user else "assistant", "content": message}
                for message, is_user in messages
            ]
        )
        
        self._update_records(model_name, response)
        
        money_spent = self.calc_money_spent(model_name, response)
        total_money_spent = self.money_spent['total']

        logging.info(f'{model_name} costs ${money_spent: .2e} on this invocation; cumulative total cost from all models: ${total_money_spent: .2e}')
        
        stop_reason = response['choices'][0]['finish_reason']
        if stop_reason != 'stop':
            message_length = sum(len(message) for message, _ in messages)
            error_message = f'Text generation by {model_name} stopped abnormally! Stop reason: {stop_reason}; total input messages length: {message_length}'
            
            logging.warning(error_message)
            
            raise Exception(error_message)
        else:
            result = response['choices'][0]['message']['content']
            logging.debug(f'Response from {model_name}:\n<response-start>\n{result}\n<response-end>')

            return result
    
    @property
    def money_spent(self) -> Dict[str, float]:
        """
        the money spent, in dollars.
        """
        
        money_spent_by_model = {}

        for model_name in self.price_info.keys():
            input_price, output_price = self.price_info[model_name]
            input_tokens, output_tokens = self.token_usages[model_name]
            
            money_spent_by_model[model_name] = input_price * input_tokens / 1000 + output_price * output_tokens / 1000

        money_spent_by_model['total'] = sum(money_spent_by_model.values())

        return money_spent_by_model
    
    def calc_money_spent(self, model_name: str, response: Dict) -> float:
        token_usage = response['usage']
        completion_tokens, prompt_tokens = (
            token_usage['completion_tokens'],
            token_usage['prompt_tokens']
        )
        
        return self.price_info[model_name][0] * prompt_tokens / 1000 + self.price_info[model_name][1] * completion_tokens / 1000
            
    def _update_records(self, model_name: str, response: Dict) -> None:
        token_usage = response['usage']
        completion_tokens, prompt_tokens = (
            token_usage['completion_tokens'],
            token_usage['prompt_tokens']
        )
        
        current_total_input_tokens, current_total_output_tokens = self.token_usages[model_name]
        self.token_usages[model_name] = (current_total_input_tokens + prompt_tokens, current_total_output_tokens + completion_tokens)
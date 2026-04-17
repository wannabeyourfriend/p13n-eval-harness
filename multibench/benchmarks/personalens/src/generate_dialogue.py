# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

import os
import json
import logging
try:
    import boto3
    from botocore.exceptions import ClientError
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False
import diskcache
from os import path
from time import sleep
from typing import List, Dict
import argparse
from ..util.assistant_prompts import *

BEDROCK_SERVICE = "bedrock-runtime"

# vLLM / OpenAI-compatible API support
VLLM_API_BASE = os.environ.get("VLLM_API_BASE", "http://localhost:8000/v1")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "not-needed")
VLLM_MODEL_NAME = os.environ.get("VLLM_MODEL_NAME", "default")

model_id_dict = {
    "anthropic.claude-3-5-sonnet-20240620-v1:0": "claude-3-5-sonnet-v1",
    "us.anthropic.claude-3-opus-20240229-v1:0": "claude-3-opus-v1",
    "us.anthropic.claude-3-5-sonnet-20240620-v1:0": "claude-3-5-sonnet-v1",
    "anthropic.claude-3-sonnet-20240229-v1:0": "claude-3-sonnet-v1",
    "us.anthropic.claude-3-haiku-20240307-v1:0": "claude-3-haiku-v1",
    "us.anthropic.claude-3-5-haiku-20241022-v1:0": "claude-3-5-haiku-v1",
    "us.anthropic.claude-3-5-sonnet-20241022-v2:0": "claude-3-5-sonnet-v2",
    "us.anthropic.claude-3-7-sonnet-20250219-v1:0": "claude-3-7-sonnet-v1",
    "meta.llama3-70b-instruct-v1:0": "llama-3-70b-instruct-v1",
    "us.meta.llama3-1-70b-instruct-v1:0": "llama-3-1-70b-instruct-v1",
    "us.meta.llama3-2-90b-instruct-v1:0": "llama-3-2-90b-instruct-v1",
    "us.meta.llama3-3-70b-instruct-v1:0": "llama-3-3-70b-instruct-v1",
    "us.meta.llama3-1-8b-instruct-v1:0": "llama-3-1-8b-instruct-v1",
    "mistral.mistral-7b-instruct-v0:2": "mistral-7b-instruct-v2",
    "mistral.mixtral-8x7b-instruct-v0:1": "mixtral-8x7b-instruct-v1"
}

model_id_reverse_dict = {
    "claude-3-5-sonnet-v1": "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "claude-3-sonnet-v1": "anthropic.claude-3-sonnet-20240229-v1:0",
    "claude-3-haiku-v1": "us.anthropic.claude-3-haiku-20240307-v1:0",
    "claude-3-5-haiku-v1": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    "claude-3-5-sonnet-v2": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    "claude-3-7-sonnet-v1": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    "llama-3-70b-instruct-v1": "meta.llama3-70b-instruct-v1:0",
    "llama-3-1-70b-instruct-v1": "us.meta.llama3-1-70b-instruct-v1:0",
    "llama-3-3-70b-instruct-v1": "us.meta.llama3-3-70b-instruct-v1:0",
    "llama-3-1-8b-instruct-v1": "us.meta.llama3-1-8b-instruct-v1:0",
    "mistral-7b-instruct-v2": "mistral.mistral-7b-instruct-v0:2",
    "mixtral-8x7b-instruct-v1": "mistral.mixtral-8x7b-instruct-v0:1"
}

if "DATA_DIR" in os.environ:
    DATA_DIR = os.environ["DATA_DIR"]
else:
    DATA_DIR = path.join(os.getenv("HOME"), 'workspace', 'data')

CACHES_DIR = path.join(DATA_DIR, 'caches')


def _strip_think_tags(text):
    """Remove Qwen3-style <think>...</think> reasoning blocks."""
    import re
    text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
    text = re.sub(r'</think>\s*', '', text)
    return text.strip()


class VllmLLM:
    """LLM client using OpenAI-compatible API (vLLM server)."""
    def __init__(self, model_name=None, api_base=None, api_key=None, max_tokens=512,
                 temperature=0.5, system_prompt="") -> None:
        from openai import OpenAI
        self.model_name = model_name or VLLM_MODEL_NAME
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.client = OpenAI(
            base_url=api_base or VLLM_API_BASE,
            api_key=api_key or VLLM_API_KEY,
        )
        self.cache = diskcache.Cache(path.join(CACHES_DIR, f"vllm_{self.model_name}"))

    def invoke(self, messages, use_caching=True):
        """messages can be a string (single user turn) or list of dicts."""
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        cache_key = json.dumps(messages, sort_keys=True)
        if cache_key in self.cache and use_caching:
            return self.cache[cache_key]

        api_messages = []
        if self.system_prompt:
            api_messages.append({"role": "system", "content": self.system_prompt})
        api_messages.extend(messages)

        backoff_time = 0.5
        for attempt in range(10):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=api_messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                completion = _strip_think_tags(response.choices[0].message.content.strip())
                self.cache[cache_key] = completion
                return completion
            except Exception as e:
                logging.warning(f"[VllmLLM] Attempt {attempt+1} failed: {e}")
                sleep(backoff_time)
                backoff_time *= 2
        raise RuntimeError(f"VllmLLM failed after 10 retries")

    def single_turn_request(self, messages):
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        return self.invoke(messages, use_caching=False)


class ClaudeLLM:
    def __init__(self, model_id='anthropic.claude-3-sonnet-20240229-v1:0', region='us-east-1', max_tokens=512,
                 temperature=0.5, system_prompt="") -> None:
        self.model_id = model_id
        self.region = region
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.cache = diskcache.Cache(path.join(CACHES_DIR, model_id))
        self.__init_session()

    def __init_session(self):
        self.session = boto3.Session()
        self.client = self.session.client(
            BEDROCK_SERVICE,
            region_name=self.region
        )

    def __invoke(self, messages):
        backoff_time = 0.5
        while True:
            try:
                # Convert messages to the format expected by Claude
                formatted_messages = [
                    {
                        "role": msg["role"],
                        "content": [
                            {
                                "type": "text",
                                "text": msg["content"]
                            }
                        ]
                    }
                    for msg in messages
                ]

                body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "system": self.system_prompt,
                    "messages": formatted_messages
                }

                return self.client.invoke_model(
                    body=json.dumps(body),
                    modelId=self.model_id
                )
            except ClientError as e:
                code = e.response['Error']['Code']
                if code == 'ThrottlingException':
                    logging.warning(f"[ThrottlingException] Waiting for ({backoff_time:.1f})s.")
                    sleep(backoff_time)
                    backoff_time *= 2
                elif code == 'ExpiredTokenException':
                    logging.warning(f"[ExpiredTokenException] Refreshing session security token.")
                elif code == 'ModelErrorException':
                    logging.warning(f'error calling {self.model_id}, trying again')
                    return self.__invoke(messages)
                elif code == 'ServiceUnavailableException':
                    logging.warning(f'Service Unavailable error calling {self.model_id}, trying again')
                    sleep(backoff_time)
                    backoff_time *= 2
                    return self.__invoke(messages)
                elif code == 'ModelTimeoutException':
                    logging.warning(f'Model timeout error calling {self.model_id}, trying again')
                    sleep(backoff_time)
                    backoff_time *= 2
                    return self.__invoke(messages)
                else:
                    raise Exception(f"Unhandled boto ClientError: {code}")

    def invoke(self, messages, use_caching=True):
        # Create a cache key from the messages
        cache_key = json.dumps(messages, sort_keys=True)
        
        if cache_key in self.cache and use_caching:
            completion = self.cache[cache_key]
        else:
            response = self.__invoke(messages)
            completion = json.loads(response.get('body').read())["content"][0]["text"]
            self.cache[cache_key] = completion
        return completion

    def single_turn_request(self, messages):
        if isinstance(messages, str):
            # Convert single string to proper message format
            messages = [{"role": "user", "content": messages}]
        return self.invoke(messages, use_caching=False)

class LlamaLLM:
    def __init__(self, model_id='meta.llama2-13b-chat-v1', region='us-east-1', max_tokens=512,
                 temperature=0.5, top_p=0.9, system_prompt="") -> None:
        self.model_id = model_id
        self.region = region
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.system_prompt = system_prompt
        self.cache = diskcache.Cache(path.join(CACHES_DIR, model_id))
        self.__init_session()

    def __init_session(self):
        self.session = boto3.Session()
        self.client = self.session.client(
            BEDROCK_SERVICE,
            region_name=self.region
        )

    def __format_messages(self, messages):
        """Format messages into Llama's expected prompt format"""
        formatted_prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        formatted_prompt += self.system_prompt if self.system_prompt else """
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  
Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
Please ensure that your responses are socially unbiased and positive in nature."""
        formatted_prompt += "<|eot_id|>\n"
        formatted_prompt += "<|start_header_id|>user<|end_header_id|>\n\n"
        formatted_prompt += messages
        formatted_prompt += "<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"
        return formatted_prompt

    def __invoke(self, messages):
        backoff_time = 0.5
        while True:
            try:
                formatted_prompt = self.__format_messages(messages)
                body = {
                    "prompt": formatted_prompt,
                    "max_gen_len": self.max_tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p
                }

                return self.client.invoke_model(
                    body=json.dumps(body),
                    modelId=self.model_id
                )
            except ClientError as e:
                code = e.response['Error']['Code']
                if code == 'ThrottlingException':
                    logging.warning(f"[ThrottlingException] Waiting for ({backoff_time:.1f})s.")
                    sleep(backoff_time)
                    backoff_time *= 2
                elif code == 'ExpiredTokenException':
                    logging.warning(f"[ExpiredTokenException] Refreshing session security token.")
                elif code == 'ModelErrorException':
                    logging.warning(f'error calling {self.model_id}, trying again')
                    return self.__invoke(messages)
                elif code == 'ServiceUnavailableException':
                    logging.warning(f'Service Unavailable error calling {self.model_id}, trying again')
                    sleep(backoff_time)
                    backoff_time *= 2
                    return self.__invoke(messages)
                elif code == 'ModelTimeoutException':
                    logging.warning(f'Model timeout error calling {self.model_id}, trying again')
                    sleep(backoff_time)
                    backoff_time *= 2
                    return self.__invoke(messages)
                else:
                    raise Exception(f"Unhandled boto ClientError: {code}")

    def invoke(self, messages, use_caching=True):
        # Create a cache key from the messages
        cache_key = json.dumps([{"role": "user", "content": messages}], sort_keys=True)
        
        if cache_key in self.cache and use_caching:
            completion = self.cache[cache_key]
        else:
            response = self.__invoke(messages)
            response_body = json.loads(response.get('body').read())
            completion = response_body['generation']
            self.cache[cache_key] = completion

        return completion

    def single_turn_request(self, messages):
        return self.invoke(messages, use_caching=False).lstrip()

class MistralLLM:
    def __init__(self, model_id='mistral.mistral-7b-instruct-v0:2', region='us-east-1', max_tokens=512,
                 temperature=0.7, top_p=1.0, system_prompt="") -> None:
        self.model_id = model_id
        self.region = region
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.system_prompt = system_prompt
        self.cache = diskcache.Cache(path.join(CACHES_DIR, model_id))
        self.__init_session()

    def __init_session(self):
        self.session = boto3.Session()
        self.client = self.session.client(
            BEDROCK_SERVICE,
            region_name=self.region
        )

    def __format_messages(self, messages):
        """Format messages into Llama's expected prompt format"""
        formatted_prompt = "<s>[INST]"
        formatted_prompt += self.system_prompt if self.system_prompt else """
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  
Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
Please ensure that your responses are socially unbiased and positive in nature.\n\n"""
        formatted_prompt += messages
        formatted_prompt += "[/INST]"
        return formatted_prompt

    def __invoke(self, messages):
        backoff_time = 0.5
        while True:
            try:
                formatted_prompt = self.__format_messages(messages)
                body = {
                    "prompt": formatted_prompt,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p
                }

                return self.client.invoke_model(
                    body=json.dumps(body),
                    modelId=self.model_id
                )
            except ClientError as e:
                code = e.response['Error']['Code']
                if code == 'ThrottlingException':
                    logging.warning(f"[ThrottlingException] Waiting for ({backoff_time:.1f})s.")
                    sleep(backoff_time)
                    backoff_time *= 2
                elif code == 'ExpiredTokenException':
                    logging.warning(f"[ExpiredTokenException] Refreshing session security token.")
                elif code == 'ModelErrorException':
                    logging.warning(f'error calling {self.model_id}, trying again')
                    return self.__invoke(messages)
                elif code == 'ServiceUnavailableException':
                    logging.warning(f'Service Unavailable error calling {self.model_id}, trying again')
                    sleep(backoff_time)
                    backoff_time *= 2
                    return self.__invoke(messages)
                elif code == 'ModelTimeoutException':
                    logging.warning(f'Model timeout error calling {self.model_id}, trying again')
                    sleep(backoff_time)
                    backoff_time *= 2
                    return self.__invoke(messages)
                else:
                    raise Exception(f"Unhandled boto ClientError: {code}")

    def invoke(self, messages, use_caching=True):
        # Create a cache key from the messages
        cache_key = json.dumps([{"role": "user", "content": messages}], sort_keys=True)
        
        if cache_key in self.cache and use_caching:
            completion = self.cache[cache_key]
        else:
            response = self.__invoke(messages)
            response_body = json.loads(response.get('body').read())
            completion = response_body['outputs'][0]['text']
            self.cache[cache_key] = completion

        return completion

    def single_turn_request(self, messages):
        return self.invoke(messages, use_caching=False).lstrip()

class ConversationSimulator:
    def __init__(self, user_llm, assistant_llm, user_prompt, assistant_prompt):
        """
        Initialize with two separate LLM models.
        
        Args:
            user_llm: LLM model instance for simulating user responses
            assistant_llm: LLM model instance for simulating assistant responses
        """
        self.user_llm = user_llm
        self.assistant_llm = assistant_llm
        # Initialize prompt templates
        self.user_prompt = user_prompt
        self.assistant_prompt = assistant_prompt
    
    def generate_initial_query(self, 
                             task_description: str,
                             demographic_profile: dict,
                             user_affinity: dict,
                             interaction_summary: str,
                             situation_context: dict) -> str:
        """Generate the initial user query based on the task description and user context."""
        # Create initial empty message history
        initial_messages = []
        
        # Format the initial prompt using the user template
        initial_prompt = self.user_prompt.format_prompt(
            task_description=task_description,
            demographic_profile=demographic_profile,
            user_affinity=user_affinity,
            interaction_summary=interaction_summary,
            situation_context=situation_context,
            message_history=initial_messages,
            initial_query=True
        )
        
        return self.user_llm.single_turn_request(initial_prompt)

    def run_user_simulation(self,
                          message_history: List[Dict[str, str]],
                          task_description: str,
                          demographic_profile: dict,
                          user_affinity: dict,
                          interaction_summary: str,
                          situation_context: dict) -> str:
        """Simulate the user's response using the user LLM."""
        # Format prompt using the user template
        prompt = self.user_prompt.format_prompt(
            task_description=task_description,
            demographic_profile=demographic_profile,
            user_affinity=user_affinity,
            interaction_summary=interaction_summary,
            situation_context=situation_context,
            message_history=message_history
        )
        
        return self.user_llm.single_turn_request(prompt)
    
    def run_assistant_simulation(self,
                                 demographic_profile: dict,
                                 interaction_summary: str,
                                 situation_context: dict,
                                 flags: list,
                                 message_history: List[Dict[str, str]]) -> str:
        """Simulate the assistant's response using the assistant LLM."""
        # Format prompt using the assistant template
        prompt = self.assistant_prompt.format_prompt(
            demographic_profile,
            interaction_summary,
            situation_context,
            flags,
            message_history=message_history
        )
        
        return self.assistant_llm.single_turn_request(prompt)

    def simulate_conversation(self,
                            task_description: str,
                            demographic_profile: dict,
                            user_affinity: dict,
                            interaction_summary: str,
                            situation_context: dict,
                            flags: list,
                            max_turns: int = 20) -> List[Dict[str, str]]:
        """Run the full conversation simulation."""
        # Generate initial query from the user LLM
        initial_query = self.generate_initial_query(
            task_description,
            demographic_profile,
            user_affinity,
            interaction_summary,
            situation_context
        )
        
        message_history = [{"role": "user", "content": initial_query}]
        print(f"User (Initial Query): {initial_query}\n")
        
        for _ in range(max_turns):
            # Get assistant's response
            assistant_response = self.run_assistant_simulation(
                demographic_profile,
                interaction_summary,
                situation_context,
                flags,
                message_history
            )
            message_history.append({"role": "assistant", "content": assistant_response})
            print(f"Assistant: {assistant_response}\n")
            
            # Get user's response
            user_response = self.run_user_simulation(
                message_history,
                task_description,
                demographic_profile,
                user_affinity,
                interaction_summary,
                situation_context
            )
            print(f"User: {user_response}\n")
            
           # Add user response to history, with or without TERMINATE
            message_history.append({"role": "user", "content": user_response})
            
            # Check if conversation should end
            if "TERMINATE" in user_response:
                # Remove TERMINATE from the last message for cleaner history
                cleaned_response = user_response.replace("TERMINATE", "").strip()
                message_history[-1]["content"] = cleaned_response
                
                # Get final assistant response
                final_assistant_response = self.run_assistant_simulation(
                    demographic_profile,
                    interaction_summary,
                    situation_context,
                    flags,
                    message_history
                )
                message_history.append({"role": "assistant", "content": final_assistant_response})
                print(f"Assistant (Final): {final_assistant_response}\n")
                break
    
        return message_history

class UserPromptTemplate:
    def __init__(self):
        self.system_prompt = """<instruction>
You are tasked with generating realistic user responses in a conversation with a virtual assistant. Your responses should follow these guidelines:

<guidelines>
- Be natural and conversational, avoiding artificial or robotic language
- Reflect the user's demographic profile and preferences provided in the <user_profile>
- Consider the <past_interaction_history> and <current_context>
- Stay consistent with the user's personality throughout the conversation
- Keep each response focused and concise (1-3 sentences maximum)
- Subtly convey your background and preferences through language
- Use English as your language
- Output 'TERMINATE' only when the task is fully completed to your satisfaction
</guidelines>

Remember: You are not an assistant - you are the user seeking help. Maintain this perspective throughout the conversation.
</instruction>"""
        
        self.task_prompt_initial_query = """<user_profile>
Demographic Information:
{demographic_profile}

User Preferences and Affinities:
{user_affinity}
</user_profile>

<past_interaction_history>
{interaction_summary}
</past_interaction_history>

<current_context>
{situation_context}
</current_context>

<task_description>
{task_description}
</task_description>

<initial_query_instructions>
Based on the above information, provide your initial query as the user. Your query should:
1. Account for your current situation
2. Be natural and conversational
3. Short and concise (1-2 sentences maximum)
4. Avoid stating specific preferences or providing excessive background information.
IMPORTANT - Do not output TERMINATE for this initial query. Output your query in English language.
</initial_query_instructions>

<examples>
"What events are happening in Basel this weekend?"
"Are there any lectures I could attend nearby?"
"Can you suggest some activities happening in the city this week?"
</examples>

<initial_query>
Your initial query:
</initial_query>"""
        

        self.task_prompt = """<user_profile>
Demographic Information:
{demographic_profile}

User Preferences and Affinities:
{user_affinity}
</user_profile>

<task_description>
{task_description}
</task_description>

<past_interaction_history>
{interaction_summary}
</past_interaction_history>

<current_context>
{situation_context}
</current_context>

<current_interaction_history>
{message_history}
</current_interaction_history>

<response_instructions>
Based on the provided information, formulate your next response as the user, following these guidelines:

1. Ensure your response is consistent with your profile and preferences outlined in the <user_profile>.
2. Consider the <past_interaction_history> and <current_context> when crafting your response.
3. Account for the details of your <current_interaction_history> in your response.
4. Maintain a natural and conversational tone, avoiding artificial or robotic language.
5. Keep your response concise, limited to 1-3 sentences maximum.

Based on <current_interaction_history>, if you feel the task has been FULLY completed AND you are SATISFIED with the outcome, add 'TERMINATE' at the end of your response.

Example: 
USER: That's exactly what I wanted! The 7:30 PM reservation at La Maison works perfectly for our anniversary.
ASSISTANT: I'm glad to hear that! Is there anything else I can assist you with?
USER: Not really, thank you! TERMINATE
ASSISTANT: I'm glad I could help. Have a wonderful anniversary!

If the task is not yet fully completed or you have remaining concerns or requirements, continue the conversation naturally without the 'TERMINATE' statement.

IMPORTANT - Output your response in English language.
</response_instructions>

<query>
Your response:
</query>
"""

    def format_prompt(self, 
                     task_description: str,
                     demographic_profile: dict,
                     user_affinity: dict,
                     interaction_summary: str,
                     situation_context: dict,
                     message_history: list,
                     initial_query: bool = False) -> str:
        """
        Format the prompt with specific information.
        
        Parameters:
        - task_description: Detailed description of what the user wants to accomplish
        - demographic_profile: Dict containing user demographic information
        - user_affinity: Dict containing user preferences in the relevant domain
        - interaction_summary: String summarizing previous relevant interactions
        - situation_context: Dict containing current situational information
        - message_history: List of previous conversation messages
        """
        
        # Format demographic profile
        demo_str = "\n".join([f"- {k}: {v}" for k, v in demographic_profile.items()])
        
        # Format user preferences
        pref_str = "\n".join([f"- {k}: {', '.join(map(str, v))}" if isinstance(v, list) else f"- {k}: {v}" for k, v in user_affinity.items()])
        
        # Format situation context
        situation_str = "\n".join([f"- {k}: {v}" for k, v in situation_context.items()])
        
        # Format message history
        history_str = "\n".join([f"[{msg['role'].upper()}]: {msg['content']}" 
                               for msg in message_history])
        
        if initial_query:
            # Format the complete prompt
            formatted_prompt = self.task_prompt_initial_query.format(
                task_description=task_description,
                demographic_profile=demo_str,
                user_affinity=pref_str,
                interaction_summary=interaction_summary,
                situation_context=situation_str
            )
        else:
            # Format the complete prompt
            formatted_prompt = self.task_prompt.format(
                task_description=task_description,
                demographic_profile=demo_str,
                user_affinity=pref_str,
                interaction_summary=interaction_summary,
                situation_context=situation_str,
                message_history=history_str
            )
        
        return formatted_prompt

class AssistantPromptTemplate:
    def __init__(self, system_prompt, task_prompt):
        self.system_prompt = system_prompt
        self.task_prompt = task_prompt
        
    def format_prompt(self,
                     demographic_profile: str,
                     interaction_summary: str,
                     situation_context: str,
                     flags: list,
                     message_history: list) -> str:
        """
        Format the prompt with specific information.
        
        Parameters:
        - message_history: List of previous conversation messages
        """
        
        # Format demographic profile
        demo_str = "\n".join([f"- {k}: {v}" for k, v in demographic_profile.items()])
        
        # Format situation context
        situation_str = "\n".join([f"- {k}: {v}" for k, v in situation_context.items()])

        # Format message history
        history_str = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" 
                               for msg in message_history])
        assert len(flags) == 3
        
        demographic_flag, interaction_flag, situation_flag = flags
        # Format the complete prompt
        if demographic_flag and interaction_flag and situation_flag:
            formatted_prompt = self.task_prompt.format(
                demographic_profile=demo_str,
                interaction_summary=interaction_summary,
                situation_context=situation_str,
                message_history=history_str
            )
        elif demographic_flag and not interaction_flag and not situation_flag:
            formatted_prompt = self.task_prompt.format(
                demographic_profile=demo_str,
                message_history=history_str
            )
        elif not demographic_flag and interaction_flag and not situation_flag:
            formatted_prompt = self.task_prompt.format(
                interaction_summary=interaction_summary,
                message_history=history_str
            )
        elif not demographic_flag and not interaction_flag and situation_flag:
            formatted_prompt = self.task_prompt.format(
                situation_context=situation_str,
                message_history=history_str
            )
        elif not demographic_flag and not interaction_flag and not situation_flag:
            formatted_prompt = self.task_prompt.format(
                message_history=history_str
            )
        
        return formatted_prompt

# Function to save the LLM answer to the specified path
def save_user_answer(user_id, task_id, answer, model_id, flags):
    tag = ""
    d, p, s = flags
    if d:
        tag += "_d"
    if p:
        tag += "_p"
    if s:
        tag += "_s"

    # Create the user-specific directory if it doesn't exist
    save_path = f"output/dialogue/user{user_id}/{model_id}{tag}"
    os.makedirs(save_path, exist_ok=True)
    
    # Save the answer to a file
    with open(os.path.join(save_path, f"{task_id}_dialogue.json"), "w") as f:
        json.dump(answer, f, indent=4)

def main(user_id, args):
    # Initialize prompt templates
    user_prompt = UserPromptTemplate()

    # Obtain ablation flags for using user context in assistant prompt
    flags = [args.demographic, args.past_interaction_summary, args.situation]

    # Determine which prompt style to use
    asst_id = args.model_id_asst
    if getattr(args, 'vllm', False):
        # vLLM mode: use Claude-style prompts (they work universally with chat APIs)
        asst_id = "claude-vllm"

    if "claude" in asst_id or "vllm" in asst_id:
        if args.demographic and args.past_interaction_summary and args.situation:
            assistant_prompt = AssistantPromptTemplate(system_prompt=CLAUDE_SYSTEM_PROMPT, task_prompt=CLAUDE_TASK_PROMPT_DPS)
        elif args.demographic and not args.past_interaction_summary and not args.situation:
            assistant_prompt = AssistantPromptTemplate(system_prompt=CLAUDE_SYSTEM_PROMPT, task_prompt=CLAUDE_TASK_PROMPT_D)
        elif not args.demographic and args.past_interaction_summary and not args.situation:
            assistant_prompt = AssistantPromptTemplate(system_prompt=CLAUDE_SYSTEM_PROMPT, task_prompt=CLAUDE_TASK_PROMPT_P)
        elif not args.demographic and not args.past_interaction_summary and args.situation:
            assistant_prompt = AssistantPromptTemplate(system_prompt=CLAUDE_SYSTEM_PROMPT, task_prompt=CLAUDE_TASK_PROMPT_S)
        elif not args.demographic and not args.past_interaction_summary and not args.situation:
            assistant_prompt = AssistantPromptTemplate(system_prompt=CLAUDE_SYSTEM_PROMPT, task_prompt=CLAUDE_TASK_PROMPT)
    elif "llama" in args.model_id_asst:
        if args.demographic and args.past_interaction_summary and args.situation:
            assistant_prompt = AssistantPromptTemplate(system_prompt=LLAMA_SYSTEM_PROMPT, task_prompt=LLAMA_TASK_PROMPT_DPS)
        elif args.demographic and not args.past_interaction_summary and not args.situation:
            assistant_prompt = AssistantPromptTemplate(system_prompt=LLAMA_SYSTEM_PROMPT, task_prompt=LLAMA_TASK_PROMPT_D)
        elif not args.demographic and args.past_interaction_summary and not args.situation:
            assistant_prompt = AssistantPromptTemplate(system_prompt=LLAMA_SYSTEM_PROMPT, task_prompt=LLAMA_TASK_PROMPT_P)
        elif not args.demographic and not args.past_interaction_summary and args.situation:
            assistant_prompt = AssistantPromptTemplate(system_prompt=LLAMA_SYSTEM_PROMPT, task_prompt=LLAMA_TASK_PROMPT_S)
        elif not args.demographic and not args.past_interaction_summary and not args.situation:
            assistant_prompt = AssistantPromptTemplate(system_prompt=LLAMA_SYSTEM_PROMPT, task_prompt=LLAMA_TASK_PROMPT)
    elif "mistral" in args.model_id_asst or "mixtral" in args.model_id_asst:
        if args.demographic and args.past_interaction_summary and args.situation:
            assistant_prompt = AssistantPromptTemplate(system_prompt=MISTRAL_SYSTEM_PROMPT, task_prompt=MISTRAL_TASK_PROMPT_DPS)
        elif args.demographic and not args.past_interaction_summary and not args.situation:
            assistant_prompt = AssistantPromptTemplate(system_prompt=MISTRAL_SYSTEM_PROMPT, task_prompt=MISTRAL_TASK_PROMPT_D)
        elif not args.demographic and args.past_interaction_summary and not args.situation:
            assistant_prompt = AssistantPromptTemplate(system_prompt=MISTRAL_SYSTEM_PROMPT, task_prompt=MISTRAL_TASK_PROMPT_P)
        elif not args.demographic and not args.past_interaction_summary and args.situation:
            assistant_prompt = AssistantPromptTemplate(system_prompt=MISTRAL_SYSTEM_PROMPT, task_prompt=MISTRAL_TASK_PROMPT_S)
        elif not args.demographic and not args.past_interaction_summary and not args.situation:
            assistant_prompt = AssistantPromptTemplate(system_prompt=MISTRAL_SYSTEM_PROMPT, task_prompt=MISTRAL_TASK_PROMPT)

    # Initialize two separate LLM models with clear, single system prompts
    if getattr(args, 'vllm', False):
        # vLLM mode: use OpenAI-compatible API for both user and assistant
        user_llm = VllmLLM(
            model_name=args.vllm_model_name,
            api_base=args.vllm_api_base,
            temperature=0.5,
            max_tokens=400,
            system_prompt=user_prompt.system_prompt
        )
        assistant_llm = VllmLLM(
            model_name=args.vllm_model_name,
            api_base=args.vllm_api_base,
            temperature=0,
            max_tokens=400,
            system_prompt=assistant_prompt.system_prompt
        )
    else:
        user_llm = ClaudeLLM(
            model_id=model_id_reverse_dict[args.model_id_user],
            region=args.bedrock_region,
            temperature=0.5,
            max_tokens=400,
            system_prompt=user_prompt.system_prompt
        )

        if "claude" in args.model_id_asst:
            assistant_llm = ClaudeLLM(
            model_id=model_id_reverse_dict[args.model_id_asst],
            region=args.bedrock_region,
            temperature=0,
            max_tokens=400,
            system_prompt=assistant_prompt.system_prompt
        )
        elif "llama" in args.model_id_asst:
            assistant_llm = LlamaLLM(
            model_id=model_id_reverse_dict[args.model_id_asst],
            region=args.bedrock_region,
            temperature=0,
            max_tokens=400,
            system_prompt=assistant_prompt.system_prompt
        )
        elif "mistral" in args.model_id_asst or "mixtral" in args.model_id_asst:
            assistant_llm = MistralLLM(
            model_id=model_id_reverse_dict[args.model_id_asst],
            region=args.bedrock_region,
            temperature=0,
            max_tokens=400,
            system_prompt=assistant_prompt.system_prompt
        )
    
    
    with open(f"data/profile/user{user_id}/profile.json") as f:
        user_profile = json.load(f)
    with open(f"data/profile/user{user_id}/tasks.json") as f:
        tasks = json.load(f)
    user_profile['demographics'].pop('user_id')

    # Path prefix for idempotency check (mirrors save_user_answer logic).
    _tag = ""
    if args.demographic: _tag += "_d"
    if args.past_interaction_summary: _tag += "_p"
    if args.situation: _tag += "_s"
    _save_dir = f"output/dialogue/user{user_id}/{args.model_id_asst}{_tag}"

    for _ , task in tasks.items():
        task_id = task['task_id']

        # Skip tasks whose dialogue file already exists — supports safe resume.
        _out_path = os.path.join(_save_dir, f"{task_id}_dialogue.json")
        if os.path.exists(_out_path):
            continue

        relevant_domains = task['Relevant Domains']

        demographic_profile = user_profile['demographics']
        user_affinity = user_profile['affinities'][relevant_domains[0]]
        interaction_summary = user_profile['interactions'][relevant_domains[0]]
        situation_context = task['situations']
        task_description = task['User Intent']

        # Initialize conversation simulator with both models
        simulator = ConversationSimulator(user_llm, assistant_llm, user_prompt, assistant_prompt)
        
        # Run the simulation
        conversation_history = simulator.simulate_conversation(
            task_description=task_description,
            demographic_profile=demographic_profile,
            user_affinity=user_affinity,
            interaction_summary=interaction_summary,
            situation_context=situation_context,
            flags=flags
        )

        output = {
            "user_id": user_id,
            "task_id": task_id,
            "task_description": task_description,
            "user_model": args.model_id_asst,
            "assistant_model": args.model_id_asst,
            "dialogue": conversation_history}
        
        
        save_user_answer(user_id, task_id, output, model_id=args.model_id_asst, flags=flags)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO, datefmt="%Y-%m-%d %H:%M")
    parser = argparse.ArgumentParser(description="Generate personas for user profiles.")

    # Define arguments for start and end indices
    parser.add_argument("-s", "--start_index", type=int, default=0, help="The starting index of the user profiles.")
    parser.add_argument("-e", "--end_index", type=int, default=1499, help="The ending index of the user profiles.")
    parser.add_argument("-s3", "--sample_30", action="store_true", help="Whether to use small sample of 30 users.")
    parser.add_argument("-s5", "--sample_50", action="store_true", help="Whether to use small sample of 50 users.")
    parser.add_argument("-s10", "--sample_100", action="store_true", help="Whether to use small sample of 100 users.")
    parser.add_argument("--sample_idxs", type=int, nargs='+', 
                   help="List of user indices to process.")
    parser.add_argument("-r", "--bedrock_region", type=str, default='us-east-1', help="The Bedrock region.")
    parser.add_argument("-u", "--model_id_user", type=str, default='claude-3-sonnet-v1', help="The model id of the user used in the dialogue generation.")
    parser.add_argument("-m", "--model_id_asst", type=str, default='claude-3-sonnet-v1', help="The model id of the assistant used in the dialogue generation.")
    parser.add_argument("-d", "--demographic", action="store_true", help="Whether to include demographic profile to assistant.")
    parser.add_argument("-p", "--past_interaction_summary", action="store_true", help="Whether to include past interaction summary to assistant.")
    parser.add_argument("-si", "--situation", action="store_true", help="Whether to include situational context to assistant.")
    # vLLM options
    parser.add_argument("--vllm", action="store_true", help="Use vLLM OpenAI-compatible API instead of Bedrock.")
    parser.add_argument("--vllm_model_name", type=str, default=VLLM_MODEL_NAME, help="Model name on vLLM server.")
    parser.add_argument("--vllm_api_base", type=str, default=VLLM_API_BASE, help="vLLM API base URL.")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of users to simulate concurrently (ThreadPoolExecutor workers). "
                             "Dialogue turns within a user remain sequential.")

    # Parse arguments
    args = parser.parse_args()


    sample_30_idxs = [7, 66, 132, 139, 207, 230, 251, 313, 340, 376, 386, 389, 415, 428, 597, 746, 774, 788, 822, 854, 900, 1111, 1142, 1150, 1197, 1231, 1322, 1458, 1492, 1495]
    sample_50_idxs = [7, 53, 66, 107, 132, 139, 167, 195, 207, 230, 251, 313, 340, 376, 386, 389, 415, 418, 428, 439, 517, 532, 597, 630, 641, 660, 674, 701, 746, 774, 788, 802, 822, 854, 900, 1111, 1142, 1145, 1150, 1197, 1231, 1322, 1335, 1362, 1377, 1439, 1458, 1492, 1493, 1495]
    sample_100_idxs = [7, 21, 53, 66, 86, 107, 132, 139, 157, 166, 167, 168, 195, 207, 230, 248, 251, 312, 313, 340, 352, 363, 365, 376, 386, 389, 394, 415, 418, 428, 431, 439, 470, 482, 517, 532, 597, 619, 630, 641, 657, 659, 660, 664, 674, 686, 689, 701, 744, 745, 746, 774, 788, 802, 813, 822, 838, 840, 842, 847, 854, 857, 870, 878, 880, 900, 913, 928, 942, 954, 997, 1069, 1111, 1114, 1118, 1120, 1142, 1145, 1150, 1151, 1167, 1184, 1197, 1231, 1246, 1322, 1330, 1335, 1345, 1362, 1377, 1384, 1434, 1439, 1458, 1461, 1492, 1493, 1495, 1496]
    

    assert len(sample_30_idxs) == 30
    assert len(sample_50_idxs) == 50
    assert len(sample_100_idxs) == 100
    if not args.vllm:
        assert args.model_id_asst in model_id_reverse_dict.keys(), f"{args.model_id_asst} is not supported for Assistant model id."
        assert args.model_id_user in model_id_reverse_dict.keys(), f"{args.model_id_user} is not supported for User model id."
    else:
        # In vLLM mode, use vllm_model_name as the model_id_asst for output paths
        args.model_id_asst = args.vllm_model_name


    if args.sample_30:
        indices = list(sample_30_idxs)
    elif args.sample_50:
        indices = list(sample_50_idxs)
    elif args.sample_100:
        indices = list(sample_100_idxs)
    elif args.sample_idxs:
        indices = list(args.sample_idxs)
    else:
        indices = list(range(args.start_index, args.end_index + 1))

    if args.parallel and args.parallel > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        logging.info(f"Generating dialogues for {len(indices)} users with {args.parallel} workers.")
        with ThreadPoolExecutor(max_workers=args.parallel) as ex:
            futs = {ex.submit(main, idx, args): idx for idx in indices}
            for fut in as_completed(futs):
                idx = futs[fut]
                try:
                    fut.result()
                except Exception as e:
                    logging.error(f"user{idx} failed: {e}")
    else:
        for idx in indices:
            main(idx, args)
    
    
       


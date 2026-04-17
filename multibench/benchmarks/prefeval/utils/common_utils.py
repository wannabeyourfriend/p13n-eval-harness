"""Shared helpers for PrefEval — pure prompt/data utilities.

Stripped down from upstream. The `generate_message` dispatcher and boto3
Bedrock code are gone; inference goes through multibench.client.LLMClient
instead. Helpers here are format-builders and constants only.

For downstream callers that still pass `model_type`, we always return
`"claude"` (dict-messages format) from `get_model_info`, which matches the
OpenAI chat-completion message schema.
"""

import os
import json
import yaml
import tiktoken


def count_tokens(text: str) -> int:
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(text))


def check_file_exists(save_file, total_len):
    if os.path.exists(save_file):
        with open(save_file, "r") as infile:
            already_saved_data = json.load(infile)
        if len(already_saved_data) == total_len:
            print(f"Already saved enough data of {total_len}, Skipping evaluation.")
            return True
        print("only have ", len(already_saved_data))
        return False
    return False


def print_conversation(messages):
    for message in messages:
        role = message["role"]
        content = message["content"]
        print(f"{role.capitalize()}: {content}\n")


def extract_multi_turn_conversation(multi_turn_message, turn_number=3, model_type="claude"):
    """Render the mid-conversation filler turns into `model_type`'s message format.

    `model_type` is kept for compatibility. All OpenAI-compatible backends use
    the "claude" branch (a list of role/content dicts).
    """
    message = []
    for turn in multi_turn_message:
        role = turn["role"]
        content = turn["content"]
        if model_type == "llama":
            message.append(f"<|start_header_id|>{role}<|end_header_id|>\n{content}<|eot_id|>")
        elif model_type in ("claude", "gpt", "vllm"):
            message.append({"role": role, "content": content})
        elif model_type == "mistral":
            if role == "user":
                message.append(f"[INST] {content} [/INST]")
            else:
                message.append(f"{content}</s>")
        elif model_type == "gemini":
            gemini_role = {"user": "user", "assistant": "model"}.get(role, "user")
            message.append({"role": gemini_role, "parts": [{"text": str(content)}]})
        else:
            raise ValueError(f"Invalid model_type: {model_type}")
        if len(message) == turn_number * 2:
            if role != "assistant":
                raise ValueError("The last turn must be from assistant")
            break
    assert len(message) == turn_number * 2, "The number of turns is less than the specified number"
    if model_type in ("llama", "mistral"):
        message = "".join(message)
    return message


def load_config(config_file):
    if not os.path.isabs(config_file):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_file = os.path.join(project_root, os.path.basename(config_file))
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    config["dir_path"] = os.path.dirname(os.path.abspath(config_file))
    return config


def extract_multi_turn_message(turns_data, args, model_type):
    if args.inter_turns > 0:
        multi_turn_message = []
        for turn_data in turns_data:
            multi_turn_message.extend(turn_data["conversation"])
        return (
            extract_multi_turn_conversation(multi_turn_message, args.inter_turns, model_type=model_type),
            multi_turn_message,
        )
    return "", None


def get_model_info(model_name: str):
    """Return (model_id, model_type). model_type is always 'claude' here — all
    backends go through the OpenAI-compatible LLMClient which takes the same
    dict-message schema as Claude/GPT."""
    return model_name, "claude"


ALL_TOPICS = [
    "travel_transportation", "shop_motors", "lifestyle_beauty", "travel_restaurant",
    "shop_fashion", "entertain_shows", "pet_ownership", "lifestyle_fit",
    "entertain_games", "shop_home", "lifestyle_health", "travel_activities",
    "education_learning_styles", "entertain_music_book",
    "professional_work_location_style", "education_resources", "lifestyle_dietary",
    "shop_technology", "travel_hotel", "entertain_sports",
]


COT_PROMPT = """
When answering a user's question, a good assistant should carefully consider the user's stated preferences and tailor the response accordingly.

Example 1:
User1's preference: "I have an intense phobia of bugs and insects. What outdoor activities would you recommend for the summer?"
Good assistant respnose: "Considering your phobia of bugs and insects, I'll focus on recommending activities that minimize encounters with insects. Let's explore outdoor options with fewer bugs, like beaches, pools, urban areas, well-maintained park trails, and activities during cooler times of day. I'll avoid suggesting things like camping, off-trail hiking, picnicking, or activities in dense forests. Here are some summer outdoor activities you might enjoy: spending a day at the beach swimming and playing sports, exploring outdoor malls or pedestrian areas in the city, hiking on maintained trails in parks (avoiding dense woods), attending evening outdoor concerts or movies, and enjoying water activities like kayaking or paddle boating on lakes or rivers. These options should help you enjoy the outdoors while minimizing encounters with insects."
Example 2:
User2's preference: "I'm a vegan looking for new dinner recipes. Any ideas?"
Good assistant response: "As a vegan, you'll need plant-based recipes without animal products. I'll focus on nutrient-rich, diverse ingredients to ensure balanced meals. Consider: quinoa and black bean burrito bowls, lentil and vegetable curry, roasted vegetable and hummus wrap, or zucchini noodles with avocado pesto. These options offer protein, fiber, and essential nutrients while adhering to vegan principles."
Example 3:
User3's preference: "I'm a beginner guitarist with small hands. What's a good first guitar?"
Good assistant response: "For a beginner with small hands, we'll prioritize comfort and playability. Look for guitars with narrow necks and shorter scale lengths. Consider: Fender Mustang, Ibanez Mikro, or Squier Mini Strat. These models are smaller overall, making them easier to handle. Avoid full-sized dreadnoughts or jumbo acoustics. Try these at a local music store to ensure they feel comfortable before purchasing."
Example 4:
User4's preference: "I'm on a tight budget but want to start a home gym. What equipment should I get?"
Good assistant response: "Working with a tight budget, we'll focus on versatile, cost-effective equipment for a home gym. Prioritize items that offer multiple exercise options. Consider: resistance bands, a set of adjustable dumbbells, a stability ball, and a yoga mat. These allow for a wide range of strength, flexibility, and cardio exercises without breaking the bank. Avoid expensive, single-purpose machines. As your budget allows, you can add more specialized equipment later."
Example 5:
User5's preference: "I'm sensitive to caffeine but need an energy boost in the mornings. Any suggestions?"
Good assistant response: "Given your caffeine sensitivity, we'll explore natural, caffeine-free energy boosters. Focus on lifestyle changes and alternatives to caffeinated drinks. Try: starting your day with a brisk walk or light exercise, drinking a glass of water with lemon, eating a protein-rich breakfast, or trying herbal teas like peppermint or ginger. Avoid coffee, energy drinks, and black tea. These alternatives can help increase alertness without the jitters or crashes associated with caffeine."

Now, please answer the following question while considering my preferences (not the user preferences in the examples above), which I have stated either explicitly or implicitly in our previous conversation:

            """
REMINDER = "\nIn your response, please ensure that you take into account our earlier discussion, and provide an answer that is consistent with my preference."

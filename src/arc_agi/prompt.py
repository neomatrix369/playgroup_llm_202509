from jinja2 import Environment, FileSystemLoader

# import util_image_encoder
from arc_agi import representations, utils

# shows all the contents in one line
# Please write a 1 sentence description about {{ patterns }}.

# {% for pattern in patterns %}
# Here is an example input and output pattern
# {{ pattern }}
# {% endfor %}


def make_prompt(prompt_name, patterns, target="train", func_dict={}):
    """
    Create a prompt from a file-based template plus patterns.
    """
    environment = Environment(loader=FileSystemLoader("prompts/"))
    template = environment.get_template(prompt_name)
    # add any functions to the template
    template.globals.update(func_dict)
    # only provide the train patters
    prompt = template.render(patterns_input_output=patterns[target])
    return prompt


def get_func_dict():
    func_dict = {
        "write_grid": representations.write_grid,
        "make_grid_plain": representations.make_grid_plain,
        "make_grid_csv_quoted": representations.make_grid_csv_quoted,
        "make_grid_csv_english_words": representations.make_grid_csv_english_words,
        "make_grid_csv": representations.make_grid_csv,
        # "base64_grid_image": util_image_encoder.save_pixel_block,
        # "base64_grid_image_href": util_image_encoder.save_pixel_block_href,
        "make_excel_description_of_example": representations.make_excel_description_of_example,
    }
    return func_dict


if __name__ == "__main__":
    parser = utils.add_argument_parser(problem_name=True, template_name=True)

    args = parser.parse_args()
    print(args)

    func_dict = get_func_dict()
    patterns = utils.get_examples(args.problem_name)
    prompt = make_prompt(args.template_name, patterns, target="train", func_dict=get_func_dict())
    print(prompt)

def load_text(file):
    with open(file, "r") as file:
        text = file.read()
    return text
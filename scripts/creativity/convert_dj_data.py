import json

paths = [
    '/home/ubuntu/creativity/data/fake_news/olmo-7b-instruct_fake_news_maxtoken164_minlength128.json',
    '/home/ubuntu/creativity/data/fake_news/olmo-7b_fake_news_maxtoken164_minlength128.json',
    '/home/ubuntu/creativity/data/letters/olmo-7b-internal_letters_maxtoken1024.json',
    # '/home/ubuntu/creativity/data/letters/olmo-7b_letters_maxtoken1024.json',
    '/home/ubuntu/creativity/data/ml_papers/olmo-7b-instruct_ml_papers_maxtoken164_minlength128.json',
    '/home/ubuntu/creativity/data/new_book/olmo-7b-internal_new_book.json',
    '/home/ubuntu/creativity/data/new_book/olmo-7b_new_book.json',
    '/home/ubuntu/creativity/data/old_man_sea/olmo-7b-internal_old_man_sea.json',
    # '/home/ubuntu/creativity/data/old_man_sea/olmo-7b_old_man_sea.json',
    '/home/ubuntu/creativity/data/poems/olmo-7b-instruct_poems_maxtoken164_minlength128.json',
    '/home/ubuntu/creativity/data/poems/olmo-7b_poems_maxtoken164_minlength128.json',
    '/home/ubuntu/creativity/data/speech/olmo-7b-instruct_speech_maxtoken164_minlength128.json',
    '/home/ubuntu/creativity/data/theorem/olmo-7b-instruct_theorem_maxtoken164_minlength128.json',
]

for path in paths:
    with open(path) as f:
        js = json.load(f)
    for i, item in enumerate(js):
        with open(f'att_input_2/{path.split("/")[-1][:-5]}.{i}', 'w') as f:
            f.write(item['text'])

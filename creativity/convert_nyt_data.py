import csv

responses = []
with open('att_input/NYTimes_correct_version (2).csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        responses.append(row['response'])

for i, response in enumerate(responses):
    with open(f'att_input_nyt/{i}', 'w') as f:
        f.write(response)

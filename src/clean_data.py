import argparse
import json

def show_datapoint(data):
    print("\n\n")
    print(f"Question: {data['question']}\n")
    context = data['context'].split('=')[1:]
    context_str = '\t- ' + ('\n\t- '.join(context))
    print(f"Context:\n{context_str}\n")

    program = data['program'].split('=')[1:]
    program = [p for p in program if not ("->" in p)]
    important_program = []
    for p in program:
        # Split p by ":" and check if the first element is "Q" and if it is then remove that character from the first element and transform the string into an int

        if p[0] == 'Q':
            idx = int(p.split(":")[0][1:])
            for q in program:
                if q[0] == 'A' and int(q.split(":")[0][1:]) == idx:
                    p += "\n\t\t" + q[q.find(":") + 1:].strip()
                    break
            important_program.append(p)
        if p[0] == 'P':
            important_program.append(p)

    program_str = '\t- ' + ('\n\t- '.join(important_program))
    print(f"Program:\n{program_str}\n")
    print(f"Answer: {data['answer']}")

def correct_data(dataset, N):
    i = 0
    while True:
        data = dataset[i]
        print("\nDATAPOINT #{}".format(i))

        show_datapoint(data)

        correct = input("\nIs this correct? (y/n): ")

        if correct == 'quit':
            break

        if correct == 'y':
            dataset[i]['correct'] = True
        else:
            dataset[i]['correct'] = False

        print("-" * 80)
        i += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='data/realFP/test_realfp.json', help='Path to the dataset')
    parser.add_argument('--corrected_dataset', type=str, default='./data/realFP/correct_test_realfp.json', help='Path to the dataset')
    parser.add_argument('-N', type=int, default=-1, help='The number of points to view')

    args = parser.parse_args()

    if (args.dataset[-5:] == '.json'):
        with open(args.dataset, 'rb') as f:
            dataset = json.load(f)

    correct_data(dataset, args.N)
    print("Done showing dataset")


import argparse
import json
import pickle
import numpy as np

CUR_DATASET_PATH = "./data/verified_data/cur_dataset.pkl"
INDICES_TO_EDIT_PATH = "./data/verified_data/indices_to_edit.pkl"
DATASET_PATH = "./data/verified_data/dataset.json"

CUR_DATASET = [160, 105, 206, 54, 466, 552, 31, 97, 436, 241]
INDICES_TO_EDIT = [466]

TEST_DATASET_PATH = "./data/realFP/test_realfp.json"
TEST_DISTRACT_DATASET_PATH = "./data/realFP/distractor_setting/test_distractor_realfp.json"

VERIFICATION_QUESTIONS = {
    160: "What is the volume of RBCs, in cc, in the blood of an adult male?",
    105: "How much farmed land does one need to survive?",
    206: "How many tons of batteries would be needed to power a car for one year?",
    241: "If all the string was removed from all of the tennis rackets in the US and layed out end-to-end, how many round trips from Detroit to Orlando could be made with the string?"
}

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

def correct_data(dataset, N, seed=100):
    with open(CUR_DATASET_PATH, 'rb') as f:
        good_indices = pickle.load(f)
    with open(INDICES_TO_EDIT_PATH, 'rb') as f:
        indices_to_edit = pickle.load(f)

    # good_indices = CUR_DATASET
    # indices_to_edit = INDICES_TO_EDIT

    length = len(dataset)
    indices = np.arange(length)
    # remove indices from good_indices
    indices = np.setdiff1d(indices, good_indices)
    print("The original dataset has {} datapoints but the dataset we are sampling from has length {}".format(length, len(indices)))

    np.random.seed(seed)
    np.random.shuffle(indices)

    i = 0
    while True:
        idx = indices[i]
        data = dataset[idx]
        print("\nDATAPOINT #{}".format(idx))

        show_datapoint(data)

        correct = input("\nIs this correct? (y/n): ")

        if correct == 'quit':
            break

        if correct == 'y':
            good_indices.append(idx)
        elif correct == 'e':
            good_indices.append(idx)
            indices_to_edit.append(idx)

        print(f"Current Dataset Length: {len(good_indices)}" + "\n" + "-" * 80)
        i += 1

    print("Saving indices to edit... {}".format(indices_to_edit))
    with open(INDICES_TO_EDIT_PATH, 'wb') as f:
        pickle.dump(indices_to_edit, f)
    print("Saving current dataset... {}".format(good_indices))
    with open(CUR_DATASET_PATH, 'wb') as f:
        pickle.dump(good_indices, f)

def build_dataset():
    with open(CUR_DATASET_PATH, 'rb') as f:
        good_indices = pickle.load(f)
    with open(INDICES_TO_EDIT_PATH, 'rb') as f:
        indices_to_edit = pickle.load(f)

    with open(TEST_DATASET_PATH, 'rb') as f:
        test_dataset = json.load(f)

    with open(TEST_DISTRACT_DATASET_PATH, 'rb') as f:
        test_distract_dataset = json.load(f)

    dataset = []
    for idx in good_indices:
        data = test_dataset[idx]
        data["distract_context"] = test_distract_dataset[idx]["context"]
        show_datapoint(data)
        units = input("\nWhat are the units for the solution?: ")
        data["units"] = units
        if idx in indices_to_edit:
            correct_ans = input("\nWhat is the correct answer?: ")
            data["answer"] = correct_ans
        try:
            data["answer"] = np.float64(data["answer"].split()[0])
        except:
            data["answer"] = np.float64(input("What is the numerical answer?: "))
        dataset.append(data)

    with open(DATASET_PATH, 'w') as f:
        json.dump(dataset, f)



if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', type=str, default='data/realFP/test_realfp.json', help='Path to the dataset')
    # parser.add_argument('--corrected_dataset', type=str, default='./data/realFP/correct_test_realfp.json', help='Path to the dataset')
    # parser.add_argument('-N', type=int, default=-1, help='The number of points to view')

    # args = parser.parse_args()

    # if (args.dataset[-5:] == '.json'):
    #     with open(args.dataset, 'rb') as f:
    #         dataset = json.load(f)

    # correct_data(dataset, args.N)
    build_dataset()
    print("DONE!!!")
    # print("Done showing dataset")


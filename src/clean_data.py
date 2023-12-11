import argparse
import json
import os
import pickle
import numpy as np

CUR_DATASET_PATH = "./data/verified_data/cur_dataset.pkl"
INDICES_TO_EDIT_PATH = "./data/verified_data/indices_to_edit.pkl"
DATASET_DIR = "./data/verified_data/"



TEST_DATASET_PATH = "./data/realFP/test_realfp.json"
TEST_DISTRACT_DATASET_PATH = "./data/realFP/distractor_setting/test_distractor_realfp.json"

GOOD_INDICES = [160, 105, 206, 54, 466]
CUR_DATASET = [160, 105, 206, 54, 466, 552, 31, 97, 436, 241]
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

def correct_data(dataset, distract_dataset, seed=100):
    with open(CUR_DATASET_PATH, 'rb') as f:
        dataset_indices = pickle.load(f)
        # good_indices = GOOD_INDICES
    length = len(dataset)
    indices = np.arange(length)
    indices = np.setdiff1d(indices, dataset_indices)
    print("The OG dataset has {} datapoints and the new dataset has length: {}".format(length, len(dataset_indices)))

    # np.random.seed(seed)
    np.random.shuffle(indices)
    indices = list(indices)

    i = 0

    new_dataset = []

    while True:
        idx = indices[i]
        i += 1
        data = dataset[idx]
        entry = data
        print("\nDATAPOINT #{}".format(idx))

        show_datapoint(data)

        correct = input("\nIs this correct? [yes (y)/no (n)/edit (e)]: ")

        if correct == 'quit':
            break

        if correct == 'n':
            continue
        elif correct == 'e':
            entry["need_to_edit"] = True
            # correct_question = input("\nWhat is the corrected question?: ")
            # entry["question"] = correct_question
            # correct_ans = input("\nWhat is the corrected answer?: ")
            # entry["answer"] = correct_ans
        else:
            entry["need_to_edit"] = False

        try:
            entry["answer"] = np.float64(data["answer"].split()[0])
        except:
            entry["answer"] = np.float64(input("What is the numerical answer?: "))
        units = input("\nWhat are the units for the solution?: ")
        entry["units"] = units
        if (distract_dataset[idx]["question"] != entry["question"]) and (correct != 'e'):
            print("Distract question does not match")
            continue
        entry["distractor_context"] = distract_dataset[idx]["context"]
        keys = ["space", "time", "causal", "common_sense", "science"]
        reasoning = {k: False for k in keys}
        # reasoning["spatial"] = input("Does this question require spatial reasoning?: ") in ['y', 'yes', 'Y', 'Yes', 'YES']
        # reasoning["temporal"] = input("Does this question require temporal reasoning?: ") in ['y', 'yes', 'Y', 'Yes', 'YES']
        # reasoning["causal"] = input("Does this question require cause and effect reasoning?: ") in ['y', 'yes', 'Y', 'Yes', 'YES']
        # reasoning["commonsense"] = input("Does this question require common sense reasoning?: ") in ['y', 'yes', 'Y', 'Yes', 'YES']
        # reasoning["science"] = input("Does this question require scientific knowledge?: ") in ['y', 'yes', 'Y', 'Yes', 'YES']
        entry["reasoning"] = reasoning
        entry["human_verified"] = False

        new_dataset.append(entry)
        dataset_indices.append(idx)

        print(f"Current Dataset Length: {len(dataset_indices)}" + "\n" + "-" * 80)

    print("Saving dataset indices ... {}".format(dataset_indices))
    with open(CUR_DATASET_PATH, 'wb') as f:
        pickle.dump(dataset_indices, f)

    print("Saving current dataset... ")
    # with open(DATASET_PATH, 'r') as f:
    #     old_dataset = json.load(f)
    dataset = new_dataset
    with open(os.path.join(DATASET_DIR, f"dataset-len{len(dataset_indices)}.json"), 'w') as f:
        json.dump(dataset, f)

# def build_dataset():
#     with open(CUR_DATASET_PATH, 'rb') as f:
#         good_indices = pickle.load(f)
#     with open(INDICES_TO_EDIT_PATH, 'rb') as f:
#         indices_to_edit = pickle.load(f)


#     with open(TEST_DATASET_PATH, 'rb') as f:ls
#         test_dataset = json.load(f)

#     with open(TEST_DISTRACT_DATASET_PATH, 'rb') as f:
#         test_distract_dataset = json.load(f)

#     dataset = []
#     for idx in good_indices:
#         data = test_dataset[idx]
#         data["distract_context"] = test_distract_dataset[idx]["context"]
#         show_datapoint(data)
#         units = input("\nWhat are the units for the solution?: ")
#         data["units"] = units
#         if idx in indices_to_edit:
#             correct_ans = input("\nWhat is the correct answer?: ")
#             data["answer"] = correct_ans
#         try:
#             data["answer"] = np.float64(data["answer"].split()[0])
#         except:
#             data["answer"] = np.float64(input("What is the numerical answer?: "))
#         dataset.append(data)

#     with open(DATASET_PATH, 'wr') as f:
#         init_dataset = json.load(f)
#         dataset = init_dataset + dataset
#         json.dump(dataset, f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='data/realFP/test_realfp.json', help='Path to the dataset')
    parser.add_argument('--corrected_dataset', type=str, default='./data/realFP/correct_test_realfp.json', help='Path to the dataset')
    parser.add_argument('--distract_dataset', type=str, default='data/realFP/distractor_setting/test_distractor_realfp.json', help='Path to the dataset')
    args = parser.parse_args()

    # if (args.dataset[-5:] == '.json'):
    with open(args.dataset, 'rb') as f:
        dataset = json.load(f)

    with open(args.distract_dataset, 'rb') as f:
        distract_dataset = json.load(f)

    correct_data(dataset, distract_dataset)
    # build_dataset()
    print("DONE!!!")
    # print("Done showing dataset")




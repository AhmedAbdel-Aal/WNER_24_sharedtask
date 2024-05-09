from torch.utils.data import DataLoader
from torchtext.vocab import vocab
from collections import Counter, namedtuple
import re
import itertools
from utils_helper import load_object
from datasets import Token



def conll_to_segments(filename):
    """
    Convert CoNLL files to segments. This return list of segments and each segment is
    a list of tuples (token, tag)
    :param filename: Path
    :return: list[[tuple]] - [[(token, tag), (token, tag), ...], [(token, tag), ...]]
    """
    segments, segment = list(), list()

    with open(filename, "r") as fh:
        for token in fh.read().splitlines():
            if not token.strip():
                segments.append(segment)
                segment = list()
            else:
                parts = token.split()
                

                # we know that parts are at least 2, the token and the main tag
                if len(parts) == 2:
                    token = Token(text=parts[0], main_tag=[parts[1]])
                elif len(parts) == 3:
                    token = Token(text=parts[0], main_tag=[parts[1]], l2_tags=[parts[2]])
                elif len(parts) == 4:
                    token = Token(text=parts[0], main_tag=[parts[1]], l2_tags=[parts[2]], l3_tags=parts[3:])
                else:
                    raise ValueError("Unexpected length of CoNLL format for WojoodFine dataset.")
                segment.append(token)

        segments.append(segment)

    return segments


def parse_conll_files(data_paths):
    """
    Parse CoNLL formatted files and return list of segments for each file and index
    the vocabs and tags across all data_paths
    :param data_paths: tuple(Path) - tuple of filenames
    :return: tuple( [[(token, tag), ...], [(token, tag), ...]], -> segments for data_paths[i]
                    [[(token, tag), ...], [(token, tag), ...]], -> segments for data_paths[i+1],
                    ...
                  )
             List of segments for each dataset and each segment has list of (tokens, tags)
    """
    vocabs = namedtuple("Vocab", ["tags", "tokens"])
    datasets, tags, l2_tags, l3_tags ,tokens = list(), list(), list(), list(), list()

    for data_path in data_paths:
        dataset = conll_to_segments(data_path)
        datasets.append(dataset)
        tokens += [token.text for segment in dataset for token in segment]
        tags += [token.main_tag for segment in dataset for token in segment]
        l2_tags += [token.l2_tags for segment in dataset for token in segment if token.l2_tags is not None]
        l3_tags += [token.l3_tags for segment in dataset for token in segment if token.l2_tags is not None]

    # Flatten list of tags
    tags = list(itertools.chain(*tags))
    l2_tags = list(itertools.chain(*l2_tags))
    l3_tags = list(itertools.chain(*l3_tags))

    # Generate vocabs for tags and tokens
    #tag_vocabs = tag_vocab_by_type(tags)
    #tag_vocabs.insert(0, vocab(Counter(tags)))
    #vocabs = vocabs(tokens=vocab(Counter(tokens), specials=["UNK"]), tags=tag_vocabs)
    
    all_main_tags = list(set([tag for tag in tags ]))
    all_l2_tags = list(set([tag for tag in l2_tags ]))
    all_l3_tags = list(set([tag for tag in l3_tags ]))
    print(len(all_main_tags), len(all_l2_tags), len(all_l3_tags))

    main_label_map = { v:index for index, v in enumerate(all_main_tags)}
    l2_label_map = { v:index for index, v in enumerate(all_l2_tags)}
    l3_label_map = { v:index for index, v in enumerate(all_l3_tags)}
    

    return tuple(datasets), main_label_map, l2_label_map, l3_label_map


def tag_vocab_by_type(tags):
    vocabs = list()
    c = Counter(tags)
    print(c)
    tag_names = c.keys()
    tag_types = sorted(list(set([tag.split("-", 1)[1] for tag in tag_names if "-" in tag])))
    print(tag_types)
    print(tag_names)

    for tag_type in tag_types:
        r = re.compile(".*-" + tag_type + "$")
        t = list(filter(r.match, tags)) + ["O"]
        print(tag_type, t)
        print(Counter(t))
        print('----------------')
        vocabs.append(vocab(Counter(t), specials=["<pad>"]))

    return vocabs


def text2segments(text):
    """
    Convert text to a datasets and index the tokens
    """
    dataset = [[Token(text=token, gold_tag=["O"]) for token in text.split()]]
    tokens = [token.text for segment in dataset for token in segment]

    # Generate vocabs for the tokens
    segment_vocab = vocab(Counter(tokens), specials=["UNK"])
    return dataset, segment_vocab


def get_dataloaders(
    datasets, label_map, data_config, batch_size=32, num_workers=0, shuffle=(False, False, False)
):
    """
    From the datasets generate the dataloaders
    :param datasets: list - list of the datasets, list of list of segments and tokens
    :param batch_size: int
    :param num_workers: int
    :param shuffle: boolean - to shuffle the data or not
    :return: List[torch.utils.data.DataLoader]
    """
    dataloaders = list()

    for i, examples in enumerate(datasets):
        data_config["kwargs"].update({"examples": examples, "label_map": label_map})
        dataset = load_object(data_config["fn"], data_config["kwargs"])

        dataloader = DataLoader(
            dataset=dataset,
            shuffle=shuffle[i],
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=dataset.collate_fn,
        )

        print("%s batches found", len(dataloader))
        dataloaders.append(dataloader)

    return dataloaders

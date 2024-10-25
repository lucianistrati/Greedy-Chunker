from typing import Collection, NamedTuple, Sequence, List, Tuple, Optional
from dataclasses import dataclass
from collections import Counter

import pytest


class Entity(NamedTuple):
    """Representation of an entity."""

    # type of entity
    name: str
    # start and end index of entity in some word sequence (interval end is exclusive)
    position: tuple[int, int]


@dataclass
class Chunk:
    """Chunk representation."""

    # start and end index of chunk in some word sequence (interval end is exclusive)
    position: tuple[int, int]
    # entities inside this chunk
    entities: Collection[Entity]
    

def check_chunks(res_chunks: List[str], chunks_list: List[List[str]], mode: bool) -> bool:
    """Looks for the string presence or absence of each element from the res_chunks in the chunks_list"""
    for chunks in chunks_list:
        if res_chunks == chunks:
            return mode
        
        counter = 0
        for res_chunk in res_chunks:
            for chunk in chunks:
                if res_chunk in chunk:
                    counter += 1
                    break
        if counter == len(chunks):
            return mode
    return not mode


def extract_entities(text: str, names: list[str]) -> list[tuple[int, int]]:
    """
    Finds all occurences of all the elements from names list in the given text.
    """
    entities = []
    
    # Loop through each word in the list
    for name in names:
        start = text.find(name)  # Find first occurrence
        
        # Continue finding occurrences while there are matches
        while start != -1:
            end = start + len(name)  # Calculate end index
            entities.append(Entity(name=name, position=(start, end)))  # Add the tuple (start, end) to the result
            start = text.find(name, start + 1)  # Find the next occurrence
    
    return entities


def chunkify(
    words: Sequence[str],
    entities: Collection[Entity],
    max_chunk_size: int,
    min_padding: int,
) -> list[Chunk]:
    """Split words into chunks with a maximum size and minimum padding. 
    
    Obligatory objectives:

    - Chunks must not exceed a maximum number of words max_chunk_size
    - Chunks must not be empty (i.e. always contain at least one entity)
    - Chunks must not start/end within entities
    - Each entity must have a minimum context min_padding (i.e. number of words surrounding it)
    
    Optional objective
    - Minimize the number of chunks of the given text
        
    :param words: a sequence of words
    :param entities: entities associated with words
    :param max_chunk_size: maximum number of words a chunk is allowed to contain
    :param min_padding: minimum number of words each entity must be surrounded by
    """
    for entity in entities:
        if len(entity.name.split()) > max_chunk_size:
            return None
        # If an entity has more words than a chunk can have, than that entity must be divided which contracts condition no. 3
    binary_mask = [int("]" in word or "[" in word) for word in words]
    
    num_words = len(binary_mask)
    chunk_pairs = []
    current_chunk_start = None
    indexes_history = [0]
    i = 0
    while i < num_words:
        if binary_mask[i] == 1:
            if current_chunk_start is None:
                # Start a new chunk
                current_chunk_start = max(0, i - min_padding)
            
            current_chunk_end = min(num_words, i + min_padding + 1)
            
            # Extend chunk if possible while maintaining max_chunk_size constraint
            while current_chunk_end - current_chunk_start < max_chunk_size and current_chunk_end < num_words and binary_mask[current_chunk_end] == 1:
                current_chunk_end += 1
            
            # Ensure the chunk doesn't exceed the max_chunk_size
            if current_chunk_end - current_chunk_start > max_chunk_size:
                current_chunk_end = current_chunk_start + max_chunk_size
            
            # Adjust chunk start and end to include min_padding
            chunk_start = max(0, current_chunk_start)
            chunk_end = min(num_words, current_chunk_end)
            
            # Ensure the chunk ends with at least min_padding 0s after the last 1
            while chunk_end < num_words and (chunk_end - current_chunk_start <= max_chunk_size) and (chunk_end - current_chunk_start < max_chunk_size):
                chunk_end += 1
            
            chunk_pairs.append((chunk_start, chunk_end))
            current_chunk_start = None # reset this for the next chunk
            i = chunk_end
            # move i past the current chunk
        else:
            i += 1
        indexes_history.append(i)
        
        # if this happens then there is no way to find a good chunking
        if Counter(indexes_history).most_common(1)[0][1] > 1:
            return None
    
    # Post-processing quality assurance of the obtained chunks
    binary_chunks = [binary_mask[chunk_pair[0]: chunk_pair[1]] for chunk_pair in chunk_pairs]
    
    # Test for the padding to the left and to the right of the chunk
    for chunk in binary_chunks:
        if len(chunk) > max_chunk_size or chunk.index(1) < min_padding or chunk[::-1].index(1) < min_padding or 1 not in chunk:
            return None
    
    # Obtaining the actual words maped chunks with the brackets
    bracketed_chunks = [" ".join([word for word in words[chunk_pair[0]: chunk_pair[1]]])
                        for chunk_pair in chunk_pairs]
    
    # Searching for splits in the middle of the entity by checking that the paranthesization was done right
    for chunk in bracketed_chunks:
        if chunk.count("[") != chunk.count("]"):
            return None

    def filter_entities(entities: Collection[Entity], position: Tuple[int, int]) -> Collection[Entity]:
        return [entity for entity in entities if position[0] <= entity.position[0] <= entity.position[1] <= position[1]]
    
    chunks = [Chunk(entities=filter_entities(entities, ent.position), position=ent.position)
              for ent in extract_entities(text=" ".join(words), names=bracketed_chunks)]
    
    return chunks


@pytest.mark.parametrize(
    "text, max_chunk_size, min_padding, good_chunks_list, bad_chunks_list",
    [
        ("This is a [test] sentence.",
         4,
         1,
         [["is a test sentence."]],
         [["is a test", "sentence."]]
         ),
        
        ("He reported [abdominal pain] and [fever].",
         4, 
         0,
         [["He reported abdominal pain", "and fever"],
          ["abdominal pain and fever."]],
         
           [["He reported abdominal", # chunk split in the middle of the entity
                  "pain and fever."]]            
        ),
        
        ("Yesterday she had [abdominal pain] and [fever] for an hour. [Vomiting] was not present.",
         8,
         2,
         [["she had abdominal pain and fever for an",
           "an hour. Vomiting was not present."]],
         
         [["Yesterday she had abdominal pain",  # no padding to the right
                    "and fever for an hour.",  # not enough padding
                    "Vomiting was not present."]] # no padding to the left
           
        ),
        
        ("Yesterday she had [abdominal pain] and [fever] for an hour. [Vomiting] was not present.",
         1,
         2,
         [],
         []           
        ),
        
        ("Yesterday she had [abdominal pain] and [fever] for an hour. [Vomiting] was not present.",
         3,
         1,
         [],
         []           
        ),
        
         ("Yesterday she had [abdominal pain] and [fever] for an hour. [Vomiting] was not present.",
         3,
         2,
         [],
         []           
        ),
         
        ("[abdominal pain] and [fever] for an hour. [Vomiting] was not present.",
         3,
         0,
         [['abdominal pain and', 'fever for an', 'Vomiting was not']],
         []           
        ),
        
         ("[abdominal pain] and [fever] for an hour. [Vomiting] was not present.",
         8,
         0,
         [['abdominal pain and fever for an hour. Vomiting']],
         []           
        ),
         
        ("She had [abdominal pain] and [fever] for an hour. [Vomiting] was not present.",
         8,
         3,
         [],
         []           
        ),
    ]
)
def test_chunkify(text: Sequence[str],
                  max_chunk_size: int,
                  min_padding: int,
                  good_chunks_list: List[List[str]],
                  bad_chunks_list: List[List[str]]):
    assert isinstance(max_chunk_size, int)
    assert isinstance(min_padding, int)
    assert max_chunk_size >= 1
    assert min_padding >= 0
    assert "  " not in text  # check that there is no double spacing in the text, thus .split() works well as a tokenization method
    unmark_entities = True  # this will eliminate the '[', ']' tags around the entities
    case_insensitive = True  # lowercases the text and the entities to make sure all the relevant ones are detected
    names = ["abdominal pain", "fever", "Vomiting"]
    if case_insensitive:
        text = text.lower()
        names = [name.lower() for name in names]
        bad_chunks_list = [[chunk.lower() for chunk in bad_chunks] for bad_chunks in bad_chunks_list]
        good_chunks_list = [[chunk.lower() for chunk in good_chunks] for good_chunks in good_chunks_list]
        
    words = text.split() if isinstance(text, str) else text
    assert [isinstance(words, list) and isinstance(word, str) for word in words] or isinstance(words, str)
    # Sequence can also be a set of strings as well, but in this case that would not make much sense, so we don't allow it
    # In practice the implementation is done only for words being of type 'str'
    
    names = [f"[{name}]" for name in names]
    entities = extract_entities(text, names)
    chunks: Optional[List[Chunk]] = chunkify(words=words,
                        entities=entities,
                        max_chunk_size=max_chunk_size,
                        min_padding=min_padding
    )
    if chunks:
        chunks: List[str] = [text[chunk.position[0]: chunk.position[1]] for chunk in chunks]   
        if unmark_entities:
            chunks = [chunk.replace("[", "").replace("]", "") for chunk in chunks]
    
    if good_chunks_list:
        assert check_chunks(chunks, good_chunks_list, True)
        # checks for presence in the good chunks
    if bad_chunks_list:
        assert check_chunks(chunks, bad_chunks_list, False)
        # checks for absence in the bad chunks
    if not good_chunks_list and not bad_chunks_list:
        assert chunks is None


def main():
    text = "Yesterday she had [abdominal pain] and [fever] for an hour. [Vomiting] was not present."
    names = ["abdominal pain", "fever", "Vomiting"]
    
    words = text.split() if isinstance(text, str) else text
    names = [f"[{name}]" for name in names]
    entities = extract_entities(text, names)
    
    max_chunk_size = 8
    min_padding = 2
    chunks = chunkify(words=words,
                      entities=entities,
                      max_chunk_size=max_chunk_size,
                      min_padding=min_padding)
    print(chunks)


if __name__ == "__main__":
    main()

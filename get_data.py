import numpy as np

def fetch(size, shuffle=False):
    
    tokens = []
    length = 0
    with open('tokens.txt', 'r') as file:
        for line in file:
            length += 1
            nextLine = line.strip().split(",")
            nll = min(len(nextLine), size)
            nextLine = nextLine[:nll] + [0] * (size - nll)
            tokens.append(np.array(nextLine))
    
    # tokens=np.array(tokens, dtype=np.float64)

    
    return np.array(tokens)
            

print(fetch(60)[0,:])

import numpy as np

def fetch(size, shuffle=False):
    
    tokens = []
    length = 0
    with open('tokens.txt', 'r') as file:
        for line in file:
            length += 1
            nextLine = line.strip().split(",")
            nll=len(nextLine)
            if nll>=size:
                nextLine= np.array(nextLine[:size])
            else:
                nextLine = np.array(nextLine)
                nextLine = np.append(nextLine, np.zeros(size-nll))
            tokens.append(nextLine)
    
    # tokens=np.array(tokens, dtype=np.float64)

    
    return tokens
            

print(fetch(60)[0,:])
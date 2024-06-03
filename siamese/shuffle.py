import pandas as pd

# Sample DataFrame
data = {
    'Pair': ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D'],
    'Value': [1, 2, 3, 4, 5, 6, 7, 8]
}
df = pd.DataFrame(data)
print(df)

# Split DataFrame into consecutive pairs
pairs = [df.iloc[i:i+2] for i in range(0, len(df), 2)]

# Shuffle the pairs
import random
random.shuffle(pairs)

# Merge shuffled pairs back into a single DataFrame
shuffled_df = pd.concat(pairs)

print(shuffled_df)

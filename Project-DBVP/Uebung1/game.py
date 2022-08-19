def differentFlagPermutations(X, arr):
 
    ml = arr.copy()
 
    print(" ".join(ml), end =" ")
    count = len(ml)
 
    # Traverse all possible lengths
    for z in range(X-1):
         
        # Stores all combinations
        # of length z
        tmp = []
         
        # Traverse the array
        for i in arr:
            for k in ml:
                if i not in k:
                     
                    # Generate all
                    # combinations of length z
                    tmp.append(k + i)
                    count += 1
                     
        # Print all combinations of length z
        
            
        print("\n ".join(tmp), end =" ")
         
        # Replace all combinations of length z - 1
        # with all combinations of length z
        ml = tmp
 
 
# Given array
arr = ['ا', 'ر','ش','ت','ه','س','ن']
 
# Given X
X = 5
 
differentFlagPermutations(X, arr)
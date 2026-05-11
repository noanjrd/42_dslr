def quicksort(rows, left, right ):
    if right <= left:
        return
    pivot = rows[(left+right)//2]
    
    l, r = left, right
    while l <= r:
        while rows[l] < pivot:
            l +=1
        while rows[r] > pivot:
            r -= 1
            
        if l <= r:
            rows[l], rows[r] = rows[r], rows[l]
            l +=1
            r -=1
    quicksort(rows, left, r)
    quicksort(rows, l, right)
    return rows
        
    
    
        
l = [4,88,6,11,9,8,71,3,2]
print(quicksort(l, 0, len(l)-1))
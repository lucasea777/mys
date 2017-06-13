import itertools
def ej1():
    sum(1 for _ in filter(
        lambda a: (a[0]==1 or a[1]==1 or a[2]==1) 
            or (a[1]==2 and a[2]==3) ,
            itertools.permutations(range(1,5))))
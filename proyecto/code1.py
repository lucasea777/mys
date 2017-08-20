def simulate1(working_machines=5, spare_machines=2):
    t = broken = 0
    willBeRepairedAt = float('inf')
    times_to_fail = sorted([expovariate(1) for _ in range(working_machines)])

    while True:
        if times_to_fail[0] < willBeRepairedAt:
            t = times_to_fail[0]
            broken += 1
            if broken == spare_machines + 1:
                # se rompio una y no quedan mas de repuestos
                return times_to_fail[0]
            if broken < spare_machines + 1:
                # se rompio 1 pero hay de repuesto
                # otra se va a poner a usar y se va a romper en t + exp(1)
                times_to_fail[0] += expovariate(1)
                times_to_fail.sort()
            if broken == 1:
                willBeRepairedAt = t + expovariate(8)
        else:
            t = willBeRepairedAt
            broken -= 1
            if broken > 0:
                willBeRepairedAt += expovariate(8)
            if broken == 0:
                willBeRepairedAt = float('inf')

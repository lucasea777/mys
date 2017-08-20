def simulate2(working_machines=5, spare_machines=2):
    t = broken = 0
    # hay 0 rotas
    r_times = [float('inf')]*2
    times_to_fail = sorted([expovariate(1) for _ in range(working_machines)])
    while True:
        if times_to_fail[0] < min(r_times):
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
            if broken in (1, 2) and float('inf') in r_times:
                r_times[r_times.index(float('inf'))] = t + expovariate(8)
        else:   
            broken -= 1
            t = min(r_times)
            operario = r_times.index(min(r_times))
            r_times[operario] = t + expovariate(8) if broken > 1 else float('inf')

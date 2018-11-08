import cProfile

pr = cProfile.Profile()
pr.run('Ruby_AI')
pr.disable()
pr.print_stats(sort='time')
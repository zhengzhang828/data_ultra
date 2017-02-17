﻿import optparse

def fib(n, prin):
    a, b = 0, 1
    for i in range(n):
        a, b = b, a+b
        if prin:
            print a
    return a

def Main():
    parser = optparse.OptionParser('usage %prog '+\
        '-n <fib number> -o <output file> -a (print all)',version="%prog 1.0")
   
    parser.add_option('-n', dest='num', type='int', \
        help='specify the n''th fibonacci number to output')

    parser.add_option('-o', dest='out', type='string', \
        help='specify an output file (Optional)')

    parser.add_option('-a','--all', action='store_true', dest='prin', \
        default=False, help='print all numbers up to n')

    (options, args) = parser.parse_args()

    if(options.num == None):
        print parser.usage
        exit(0)
    else:
        number = options.num

    result = fib(number, options.prin)
    print "The" + str(number) + "th fib number is " + str(result)

    if(options.out != None):
        f = open(options.out, "a")
        f.write(str(result) + "\n")

if __name__ == '__main__':
    Main()


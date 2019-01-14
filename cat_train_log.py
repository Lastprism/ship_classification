import os 
import time
import sys
def cat(delay):
	while True:
		os.system("cat train.log")
		print("############################################################################\n\n")
		time.sleep(int(delay))

if __name__ == '__main__':
	cat(sys.argv[1])
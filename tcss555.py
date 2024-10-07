# !/usr/bin/python3

import argparse
import os
import sys
import csv

def check_valid_directory(path):
	if not os.path.isdir(path):
		raise argparse.ArgumentTypeError(f"'{path}' is not a valid directory.")
	return path

def generate_output(in_dir, out_dir):
	prof_path = os.path.join(in_dir, "profile", "profile.csv")
	if not os.path.isfile(prof_path):
		print(f"{prof_path} not found, exiting")
		exit(1)
	with open(prof_path, mode='r') as f:
		reader = csv.DictReader(f)
		for row in reader:
			outstr = f"""<user
				id=\"{row['userid']}\"
				age_group=\"xx-24\"
				gender=\"female\"
				extrovert=\"3.486857895\"
				neurotic=\"2.732424\"
				agreeable=\"3.583904211\"
				conscientious=\"3.445616842\"
				open=\"3.908690526\"
			/>"""
	
			outpath = os.path.join(out_dir, row['userid'] + '.xml')
			with open(outpath, 'w+') as of:
				of.write(outstr)
	
			
	
	
if __name__ == "__main__":
	
	parser = argparse.ArgumentParser(description='Check input and output directory paths.')

	parser.add_argument('-i', '--input_dir', required=True, type=check_valid_directory,
						help='Path to the input directory')
	parser.add_argument('-o', '--output_dir', required=True, type=check_valid_directory,
						help='Path to the output directory')

	args = parser.parse_args()
	generate_output(args.input_dir, args.output_dir)

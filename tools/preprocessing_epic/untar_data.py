"""Extract tar frames from epic kitchens (rgb and flow)
"""
import sys
import os
import argparse
import tarfile


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input_dir', type=str, help='Directory of EPIC video with tar files')
    parser.add_argument(
        '--output_dir', type=str, help='Directory to store the extracted files')

    args = parser.parse_args()
    return args


def main(args):
    """Main function"""
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    for root, dirs, files in os.walk(args.input_dir):
        for f in files:
            if f.endswith('.tar'):
                input_file = os.path.join(root, f)
                extract_dir = input_file.replace(args.input_dir, args.output_dir)
                extract_dir = extract_dir.replace('.tar', '')

                if os.path.isdir(extract_dir):
                    print('{} extracted'.format(extract_dir))
                    continue

                print('Extracting to {}'.format(extract_dir))
                with tarfile.open(input_file) as my_tar:
                    def is_within_directory(directory, target):
                        
                        abs_directory = os.path.abspath(directory)
                        abs_target = os.path.abspath(target)
                    
                        prefix = os.path.commonprefix([abs_directory, abs_target])
                        
                        return prefix == abs_directory
                    
                    def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                    
                        for member in tar.getmembers():
                            member_path = os.path.join(path, member.name)
                            if not is_within_directory(path, member_path):
                                raise Exception("Attempted Path Traversal in Tar File")
                    
                        tar.extractall(path, members, numeric_owner=numeric_owner) 
                        
                    
                    safe_extract(my_tar, extract_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main(parse_args()))

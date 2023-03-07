"""Flip coordinates along the X-axis to fix the handedness issue in the PDB file."""

import os
import argparse


def main():
    """Main entry."""

    # parse input arguments
    parser = argparse.ArgumentParser(description='Flip coordinates along the X-axis')
    parser.add_argument('-i', '--pdb_fpath_src', type=str, required=True,
                        help='path to the PDB file - source')
    parser.add_argument('-o', '--pdb_fpath_dst', type=str, required=True,
                        help='path to the PDB file - destination')
    args = parser.parse_args()

    # flip coordinates along the X-axis
    os.makedirs(os.path.dirname(os.path.realpath(args.pdb_fpath_dst)), exist_ok=True)
    with open(args.pdb_fpath_src, 'r') as i_file, open(args.pdb_fpath_dst, 'w') as o_file:
        for i_line in i_file:
            cord_x = float(i_line[30:38])
            o_file.write('%s%8.3f%s' % (i_line[:30], -cord_x, i_line[38:]))

if __name__ == '__main__':
    main()

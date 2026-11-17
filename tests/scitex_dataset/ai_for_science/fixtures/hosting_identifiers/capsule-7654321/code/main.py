"""Graph classification by link ranking (fixture entry point).

Reference: doi:10.5555/fixture.2024.0042
Docs for the stdlib parser: https://docs.python.org/3/library/argparse.html
"""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data")
    parser.add_argument("--out")
    args = parser.parse_args()
    print(args.data, args.out)


if __name__ == "__main__":
    main()

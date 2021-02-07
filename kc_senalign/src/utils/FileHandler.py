class FileHandler:

    def to_file(self, filename: str, contents: list):
        with open(filename, 'w') as f:
            for line in contents:
                f.write(f'{line}\n')

    def to_binary_file(self, filename: str, contents: list):
        with open(filename, 'wb') as f:
            for line in contents:
                f.write(line)

    def from_file(self, filename: str) -> list:
        with open(filename, 'r') as f:
            contents = f.readlines()
        return contents

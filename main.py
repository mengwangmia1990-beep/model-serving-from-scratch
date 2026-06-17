from generation import generate_with_cache

def main():
    query = "where is seattle located at?"
    result = generate_with_cache(query)
    print(result)


if __name__ == "__main__":
    main()
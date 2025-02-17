- All code written must have clear and concise type annotation in function declaration
- Refrain from mutating one variable of the original type to a new type, use a new variable for the mutated version of the original variable if the mutated version have a different type
- The `utils` and `constants` packages should not call any other packages in the project to avoid import cycle, `utils` can call `constants`, `app` package can call the `api` package, and the `api` package call all other packages as it please.
  

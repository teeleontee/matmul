# matmul
Blazingly fast matrix multiplication on the GPU

## usage

```
❯ cargo run -- --help
Matrix multiplication on the GPU

Usage: rust-matmul [OPTIONS] <INPUT> <OUTPUT> <COMMAND>

Commands:
  basic   Basic implementation is just 3 loops on the host
  easy    Easy implementation is a naive implementation that uses the gpu
  medium  Medium implementation is a less naive implementation that uses local memory on the gpu
  hard    Hard is an okay implementation that optimized thread throughput
  help    Print this message or the help of the given subcommand(s)

Arguments:
  <INPUT>   Input file with the matrices that are to be multiplied
  <OUTPUT>  Output file where the result of the multiplication will be

Options:
  -l, --logs  Basic debug information
  -h, --help  Print help
```

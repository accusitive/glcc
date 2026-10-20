//! Read/eval/print loop over [`fern_repl::Repl`].

use std::io::{self, BufRead, Write};

use fern_repl::{Repl, input_incomplete, is_definition};

fn main() -> io::Result<()> {
    let mut repl = Repl::new();
    println!("fern repl (PoC). :help for commands, :quit to exit.");

    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();
    let mut buffer = String::new();

    loop {
        prompt(if buffer.is_empty() { "fern> " } else { "...>  " });
        let Some(line) = lines.next() else { break };
        let line = line?;

        if buffer.is_empty() {
            let trimmed = line.trim();
            match trimmed {
                "" => continue,
                ":quit" | ":q" => break,
                ":help" | ":h" => {
                    print_help();
                    continue;
                }
                ":clear" => {
                    repl.clear();
                    println!("cleared");
                    continue;
                }
                ":reload" | ":r" => {
                    match repl.reload() {
                        Ok(()) => println!("reloaded"),
                        Err(diags) => report(&repl, &diags),
                    }
                    continue;
                }
                _ => {}
            }
            if let Some(expr) = trimmed.strip_prefix(":e ").or_else(|| trimmed.strip_prefix(":eval ")) {
                run_expr(&mut repl, expr);
                continue;
            }
        }

        buffer.push_str(&line);
        buffer.push('\n');
        if input_incomplete(&buffer) {
            continue;
        }
        let chunk = std::mem::take(&mut buffer);
        if is_definition(&chunk) {
            match repl.define(&chunk) {
                Ok(signatures) => {
                    for line in signatures {
                        println!("{line}");
                    }
                }
                Err(diags) => report(&repl, &diags),
            }
        } else {
            run_expr(&mut repl, chunk.trim());
        }
    }
    Ok(())
}

fn run_expr(repl: &mut Repl, expr: &str) {
    match repl.eval_expr(expr) {
        Ok(value) => println!("{value}"),
        Err(diags) => report(repl, &diags),
    }
}

fn report(repl: &Repl, diags: &[fern::Diag]) {
    for diag in diags {
        eprintln!("{}", repl.render(diag));
    }
}

fn prompt(text: &str) {
    print!("{text}");
    io::stdout().flush().ok();
}

fn print_help() {
    println!(
        "\
:help            show this
:quit, :q        exit
:clear           forget all definitions
:reload, :r      re-read imported modules
:e <expr>        evaluate as an expression (even if it starts with `let`)
definitions      fun / let / data / class / instance (latest wins by name)
anything else    evaluated as an expression

shell commands
$ cmd args       run a command; yields Bool (exit success)
$(cmd)           capture stdout as String
${{expr}}          one argv element inside a command
cmd1 | cmd2      OS pipeline
x |> f           apply f to x
#env / #set / #cd  builtin namespace (also callable as #env(\"NAME\"))"
    );
}

using KITE
using Documenter

DocMeta.setdocmeta!(KITE, :DocTestSetup, :(using KITE); recursive = true)

makedocs(
    modules = [KITE],
    authors = "Sebastian Krantz",
    sitename = "KITE.jl",
    checkdocs = :exports,
    repo = Documenter.Remotes.GitHub("SebKrantz", "KITE.jl"),
    format = Documenter.HTML(
        canonical = "https://SebKrantz.github.io/KITE.jl",
        edit_link = "main",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "API reference" => "api.md",
    ],
)

deploydocs(
    repo = "github.com/SebKrantz/KITE.jl",
    devbranch = "main",
)

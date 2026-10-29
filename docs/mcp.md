# MCP server

`src/mcp_server.py` gives AI assistants access to the corpus through the
[Model Context Protocol](https://modelcontextprotocol.io/). It runs locally on the edition data in `output/` and
`site-data/` and offers these tools:

| Tool | What it returns |
|---|---|
| `corpus_stats()` | number of consilia per author and volume and the size of the vocabulary |
| `search_consilia(query, model, top_k, author_viaf, volume, lang)` | semantic search over the section embeddings, where a consilium scores with its best section |
| `list_embedding_models()` | the models `search_consilia` accepts and whether their vectors are present |
| `get_consilium(consilium_id)` | the complete consilium with title, argument, body and keywords |
| `list_keywords(query, top_k)` | terms of the keyword vocabulary (`output/vocabulary.json`) by frequency or by substring |
| `consilia_by_keyword(keyword, top_k, author_viaf, volume)` | all consilia with a keyword, with spelling suggestions when there is no exact match |
| `list_subjects(query, top_k)` | the concepts of the subject index with their counts |
| `consilia_by_subject(concept, central_only, top_k, author_viaf, volume)` | all consilia assigned to a concept, central ones first |
| `list_body_codes()` | the codes of the codebook on body and sexuality with roles and counts |
| `consilia_by_body_code(code, roles, top_k, author_viaf, volume)` | all consilia with a code, with the Latin evidence and roles |

A result of `search_consilia` gives the id, number, title, volume, author, score, the argument (up to 600
characters), the first substantial body section (up to 400 characters), the number of body sections and the
keywords. `get_consilium(id)` returns the full text.

## Embedding models

`search_consilia` uses the vectors of the search page (`hf-export/<tag>/`, listed in `site-data/models.json`) and,
when present, larger server-side vectors made by `src/gpu/embed_consilia_chunked.py` (`output/embeddings_<tag>.safetensors`
with `output/embeddings_<tag>_meta.json`). `CONSILIA_EMBED_TAG` chooses the default model, which is `qwen3-4b` when
its vectors are available. Vectors of an older data revision are not offered. The query model loads on the GPU when
there is room and otherwise on the CPU. All other tools work without embeddings.

## Connecting a client

Claude Desktop reads `claude_desktop_config.json` (on Linux `~/.config/Claude/`, on macOS
`~/Library/Application Support/Claude/`, on Windows `%APPDATA%\Claude\`), and Claude Code reads `.claude/settings.json`
in the project or `~/.claude/settings.json`. Both take the same entry:

```json
{
  "mcpServers": {
    "consilia": {
      "command": "/path/to/coding-consilia/.venv/bin/python",
      "args": ["/path/to/coding-consilia/src/mcp_server.py"],
      "cwd": "/path/to/coding-consilia",
      "env": { "CONSILIA_EMBED_TAG": "qwen3-4b" }
    }
  }
}
```

On Windows with the repository in WSL, the command is `wsl.exe` with the arguments
`["-d", "Ubuntu", "--cd", "/path/to/coding-consilia", "-e", "bash", "-lc", ".venv/bin/python src/mcp_server.py"]`.
After a restart of the client, questions such as "consilia about inheritance between illegitimate children" or "the
full text of consilium 267 in volume 4" are answered from the corpus.

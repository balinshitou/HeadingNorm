# Path-only differences from the project files

The files below differ from the versions in the authors' working tree only in the lines shown, which remove a directory of the authors' machine from the search path for third-party data. No computation is changed.

## `src/hn_paths.py`

- SHA-256 in the working tree: `825849a978c6561c0dcafaccb1df575bdb75b0b20f095f38be3f1e57994cb53b`
- SHA-256 here: `5cc4e00176c9df85cdf72459256d942190eaf92a24bbd0cf07f03040c12c18ed`

```diff
-   3. 作者本机的 ~/Documents/code-data/。
- 三处都没有时返回第 1 或第 2 处的路径（便于报错信息指出该放到哪里）。
+ 两处都没有时返回第 1 处的路径（便于报错信息指出该放到哪里）。
```

```diff
- EXTERNAL_ROOTS = ([Path(_ENV)] if _ENV else []) + [ROOT / 'external', Path('~/Documents/code-data')]
+ EXTERNAL_ROOTS = ([Path(_ENV)] if _ENV else []) + [ROOT / 'external']
```


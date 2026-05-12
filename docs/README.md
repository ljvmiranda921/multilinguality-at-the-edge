# Website

I'm not a frontend web developer so a lot of the HTML code was written by Claude (and Codex).
The color scheme was inspired by CHIA's website, font is Tomato Grotesk and Univers.
Ineractive versions of the paper's charts are based on my data (basically it just reads the same JSON / CSV file that my matplotlib plot generators read).

To preview a local version, run:

```sh
python -m http.server --directory website 8000
```

Deployment happens automatically whenever you push to the main branch.
The live version is at [https://ljvmiranda921.github.io/multilinguality-at-the-edge](https://ljvmiranda921.github.io/multilinguality-at-the-edge)

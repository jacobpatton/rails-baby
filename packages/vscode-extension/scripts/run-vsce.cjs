const cp = require("node:child_process");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { File } = require("node:buffer");

if (typeof globalThis.File === "undefined" && File) {
  globalThis.File = File;
}

const pkgDir = path.dirname(require.resolve("@vscode/vsce/package.json"));
const vsceBin = path.join(pkgDir, "vsce");
const args = process.argv.slice(2);

const extensionRoot = path.resolve(__dirname, "..");
const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), "babysitter-vsce-"));

fs.cpSync(extensionRoot, tempRoot, {
  recursive: true,
  filter: (src) => {
    const name = path.basename(src);
    return (
      name !== ".git" &&
      name !== "node_modules" &&
      !name.toLowerCase().endsWith(".vsix")
    );
  },
});

const manifestPath = path.join(tempRoot, "package.json");
const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
if (manifest.scripts && manifest.scripts["vscode:prepublish"]) {
  delete manifest.scripts["vscode:prepublish"];
}
fs.writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);

cp.execSync("npm install --omit=dev --no-audit --no-fund --workspaces=false", {
  cwd: tempRoot,
  stdio: "inherit",
});

const outFlagIndex = args.findIndex((arg) => arg === "--out" || arg === "-o");
if (outFlagIndex !== -1 && args[outFlagIndex + 1]) {
  const outPath = args[outFlagIndex + 1];
  if (!path.isAbsolute(outPath)) {
    args[outFlagIndex + 1] = path.resolve(extensionRoot, outPath);
  }
}

process.chdir(tempRoot);
process.argv = [process.execPath, vsceBin, ...args];

require(vsceBin);

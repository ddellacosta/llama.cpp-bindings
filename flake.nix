{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs";
    flake-utils.url = "github:numtide/flake-utils";
    llama-cpp.url = "github:ddellacosta/llama.cpp";
  };

  outputs = { self, nixpkgs, flake-utils, llama-cpp }:
    flake-utils.lib.eachSystem ["x86_64-linux"] (system:
      with nixpkgs.legacyPackages.${system};
      let
        t = lib.trivial;
        hl = haskell.lib;
        haskellPackages = haskell.packages.ghc945;

        name = "llama.cpp-bindings";

        project = devTools:
          let
            _ = builtins.trace (lib.attrNames llama-cpp);
            addBuildTools = (t.flip hl.addBuildTools) (devTools ++ [
              zlib
              haskellPackages.c2hs
              llama-cpp.packages.${system}.default
            ]);
          in
            builtins.trace (llama-cpp.packages.${system}.default.outPath)
            # builtins.trace (llama-cpp.outPath)
              haskellPackages.developPackage {
                # this prevents CHANGELOG/LICENSE/etc. from being found
                # root = lib.sourceFilesBySuffices ./. [ ".cabal" ".hs" ];
                root = ./.;
                name = name;
                returnShellEnv = !(devTools == []);

                modifier = (t.flip t.pipe) [
                  addBuildTools
                  hl.dontHaddock
                  # hl.appendBuildFlag "--extra-include-dirs=${llama-cpp.outPath}"
                  (drv: hl.overrideCabal drv (attrs: {
                    configureFlags = [
                      "--extra-include-dirs=${llama-cpp.outPath}/include"
                      "--extra-lib-dirs=${llama-cpp.outPath}/lib"
                    ];
                  }))
                ];
              };

      in {
        packages.pkg = project [ ]; # [3]

        defaultPackage = self.packages.${system}.pkg;

        devShell = project (with haskellPackages; [ # [4]
          cabal-fmt
          cabal-install
          haskell-language-server
          hlint
          watchexec
        ]);
      });
}

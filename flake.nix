{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs";
    flake-utils.url = "github:numtide/flake-utils";
    llama-cpp.url = "github:ggerganov/llama.cpp";
  };

  outputs = { self, nixpkgs, flake-utils, llama-cpp }:
    flake-utils.lib.eachSystem ["x86_64-linux"] (system:
      with nixpkgs.legacyPackages.${system};
      let
        t = lib.trivial;
        hl = haskell.lib;
        haskellPackages = haskell.packages.ghc945;

        name = "llama.cpp-bindings";

        llama-cpp-with-includes =
          llama-cpp.packages.${system}.default.overrideAttrs (oldAttrs: {
            postInstall = (oldAttrs.postInstall or "") + ''
              mkdir -p $out/lib
              mkdir -p $out/include
              cp *.a $out/lib/
              cp $src/*.h $out/include/
              # I don't think ghc/c2hs can process common.h because
              # it is C++, but it contains default params that I want to
              # auto-import somehow
              cp $src/examples/*.h $out/include/
            '';
          });

        project = devTools:
          let
            addBuildTools = (t.flip hl.addBuildTools) (devTools ++ [
              zlib
              haskellPackages.c2hs
              llama-cpp-with-includes
            ]);
          in
              haskellPackages.developPackage {
                root = ./.;
                name = name;
                returnShellEnv = !(devTools == []);

                modifier = (t.flip t.pipe) [
                  addBuildTools
                  hl.dontHaddock
                  hl.enableExecutableProfiling
                  (drv: hl.overrideCabal drv (attrs: {
                    configureFlags = [
                      "--ghc-options=-fprof-auto"
                      "--extra-include-dirs=${llama-cpp-with-includes}/include"
                      "--extra-lib-dirs=${llama-cpp-with-includes}/lib"
                    ];
                  }))
                ];
              };

      in {
        packages = {
          pkg = project [ ];
          default = self.packages.${system}.pkg;
          llama-cpp = llama-cpp-with-includes;
        };

        devShell = project (with haskellPackages; [
          cabal-fmt
          cabal-install
          haskell-language-server
          hlint
          watchexec
        ]);
      });
}

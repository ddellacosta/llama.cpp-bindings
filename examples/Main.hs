module Main where

import Prelude hiding (takeWhile)

import Control.Applicative ((<**>))
import Control.Exception (bracket)
import Control.Monad.IO.Class (MonadIO, liftIO)
import Control.Monad.Reader (MonadReader, ReaderT, runReaderT, asks)
import Data.Functor ((<&>), void)
import qualified Data.Vector.Storable as V
import Foreign.C.String (peekCString, withCString)
import Foreign.C.Types (CFloat, CInt)
import Foreign.ForeignPtr (newForeignPtr_)
import Foreign.Marshal.Alloc (alloca, free, malloc)
import Foreign.Marshal.Array (allocaArray, copyArray, newArray, peekArray, pokeArray, withArray)
import Foreign.Marshal.Utils (fromBool)
import Foreign.Ptr (Ptr)
import Foreign.Storable (Storable(peek, poke))
import GHC.Conc (TVar, atomically, newTVarIO, readTVar, readTVarIO, writeTVar)
import qualified LLaMACPP as L
import Options.Applicative
  ( ParseError(..)
  , Parser
  , auto
  , execParser
  , fullDesc
  , help
  , helper
  , info
  , long
  , noArgError
  , option
  , progDesc
  , short
  , showDefault
  , strOption
  , switch
  , value
  )
import Pipes (Producer, (>->), for, lift, runEffect, yield)
import Pipes.Prelude (takeWhile)

data Params = Params
  { _nCtx :: Int
  , _nThreads :: CInt
  , _nPredict :: Int
  , _nGpuLayers :: Int
  , _enableNumaOpts :: Bool
  , _prompt :: String
  , _modelPath :: String
  , _alphaFrequency :: CFloat
  , _alphaPresence :: CFloat
  , _penalizeNl :: Bool
  , _repeatPenalty :: CFloat
  , _topK :: CInt   -- <= 0 to use vocab size
  , _topP :: CFloat -- 1.0 = disabled
  , _tfsZ :: CFloat -- 1.0 = disabled
  , _typicalP :: CFloat -- 1.0 = disabled
  , _temp :: CFloat -- 1.0 = disabled
  }
  deriving (Eq, Show)

paramsParser :: Parser Params
paramsParser = Params
  <$> option auto ( long "n_ctx" <> value 512 <> showDefault <> help "context size" )
  <*> option auto ( long "n_threads" <> short 't' <> value 1 <> showDefault <> help "cpu physical cores" )
  -- the llama.cpp options set -1 to be infinite generation here,
  -- but I'd rather not use that convention...but haven't
  -- determined what I want to replace it with yet.
  <*> option auto ( long "n_predict" <> value 50 <> showDefault <> help "new tokens to predict" )
  <*> option auto (
        long "n_gpu_layers" <>
        value 0 <>
        showDefault <>
        help "number of layers to store in VRAM"
      )
  <*> switch (
        long "enable_numa_opts" <>
        help "attempt optimizations that help on some NUMA systems"
      )
  <*> strOption (
       long "prompt" <>
       short 'p' <>
       help "prompt" <>
       (noArgError . ExpectsArgError $ "you must supply a prompt")
     )
  <*> strOption (
       long "model_path" <>
       short 'm' <>
       help "path to model" <>
       (noArgError . ExpectsArgError $ "you must supply a model path")
     )
  <*> option auto ( long "alpha_freq" <> value 0.0 <> showDefault <> help "" )
  <*> option auto ( long "alpha_pres" <> value 0.0 <> showDefault <> help "" )
  <*> switch ( long "penalize_nl" <> showDefault <> help "" )
  <*> option auto ( long "repeat_penalty" <> value 1.1 <> showDefault <> help "" )
  <*> option auto ( long "top_k" <> value 40 <> showDefault <> help "" )
  <*> option auto ( long "top_p" <> value 0.95 <> showDefault <> help "" )
  <*> option auto ( long "tfs_z" <> value 1.0 <> showDefault <> help "" )
  <*> option auto ( long "typical_p" <> value 1.0 <> showDefault <> help "" )
  <*> option auto ( long "temp" <> value 0.8 <> showDefault <> help "" )

--

data Context = Context
  { _params :: Params
  , _llamaCtxParams :: Ptr L.ContextParams
  , _llamaCtx :: L.Context
  , _model :: L.Model
  , _lastNTokens :: TVar [L.Token]
  , _nPast :: TVar Int
  }

newtype ContextM a = ContextM (ReaderT Context IO a)
  deriving (Functor, Applicative, Monad, MonadIO, MonadReader Context)

runContextM :: Context -> ContextM () -> IO ()
runContextM ctx (ContextM ctxM) = runReaderT ctxM ctx


--

-- mimicking call here, somewhat: https://huggingface.co/TheBloke/open-llama-7B-v2-open-instruct-GGML#how-to-run-in-llamacpp 
-- result/bin/examples +RTS -xc -RTS -m ../models/open-llama-7b-v2-open-instruct.ggmlv3.q5_K_M.bin --n_ctx 2048 --temp 0.7 -t 8 --n_gpu_layers 32 -p "what is a good tomato recipe?"


main :: IO ()
main = do
  -- TODO:
  --   dump out build info
  --   generate default seed?

  bracket initialize cleanup $
    \(params', cpp, ctx, model') -> do

      lastNTokens' <- liftIO . newTVarIO $
        replicate (_nCtx params') 0
      nPast' <- liftIO . newTVarIO $ 0

      runContextM (Context params' cpp ctx model' lastNTokens' nPast') runLLaMA

  where
    runLLaMA :: ContextM ()
    runLLaMA = do
      liftIO . putStrLn $ "\nSystem Info:"
      liftIO $ putStrLn =<< peekCString =<< L.printSystemInfo

      params' <- asks _params
      ctx <- asks _llamaCtx

      -- tokenizing & eval

      -- "do one empty run to warm up the model"
      liftIO . allocaArray 1 $ \(tmp :: Ptr L.Token) -> do
        bos <- L.tokenBos
        pokeArray tmp [bos]
        _evalRes <- L.eval ctx tmp 1 0 (_nThreads $ params')
        L.resetTimings ctx

      let
        -- todo why is this constant here
        maxTokens = 1024
        tokenize s tks addBos =
          L.tokenize ctx s tks (fromIntegral maxTokens) (fromBool addBos)

      (tokenized, tokenizedCount) <- liftIO . allocaArray maxTokens $ \tokensPtr -> do
        tokenizedCount' <- withCString (_prompt params') $ \ts -> tokenize ts tokensPtr True

        putStrLn "\nPrompt"
        putStrLn $ _prompt params' <> "\n\n"

        putStrLn $ "\nTokenized " <> show tokenizedCount'

        putStrLn "\nRunning first eval of entire prompt"
        _evalRes <- L.eval ctx tokensPtr tokenizedCount' 0 (_nThreads params')

        (, tokenizedCount') <$>
          peekArray (fromIntegral tokenizedCount') tokensPtr

      -- update lastNTokens with the tokenized count
      lastNTokensTV <- asks _lastNTokens
      nPastTV <- asks _nPast

      liftIO $ atomically $ do
        lastNTokens' <- readTVar lastNTokensTV
        nPast' <- readTVar nPastTV
        writeTVar nPastTV $ nPast' + fromIntegral tokenizedCount
        writeTVar lastNTokensTV $
          drop (fromIntegral tokenizedCount) lastNTokens' <> tokenized

      liftIO . putStrLn $ "\nsampling"

      eos <- liftIO L.tokenEos

      -- I feel like this is not the right way to do this?
      runEffect $
        for (sample >-> takeWhile (/= eos)) $ \id' ->
          lift . liftIO $ putStr =<< peekCString =<< L.tokenToStr ctx id'


    sample :: Producer L.Token ContextM ()
    sample = do
      params' <- asks _params
      ctx <- asks _llamaCtx
      nPastTV <- asks _nPast
      nPast' <- liftIO . readTVarIO $ nPastTV
      lastNTokensTV <- asks _lastNTokens
      lastNTokens' <- liftIO . readTVarIO $ lastNTokensTV

      nVocab <- fromIntegral <$> (liftIO . L.nVocab $ ctx)

      id' <- liftIO $ sample' params' ctx nVocab lastNTokensTV

      void . liftIO . withArray [id'] $ \newTokenArrPtr ->
        L.eval ctx newTokenArrPtr 1 (fromIntegral nPast') (_nThreads $ params')

      liftIO . atomically $ do
        writeTVar nPastTV $
          if nPast' >= _nCtx params'
          then _nCtx params'
          else nPast' + 1
        writeTVar lastNTokensTV $ drop 1 lastNTokens' <> [id']

      -- liftIO $ putStr =<< peekCString =<< L.tokenToStr ctx id'

      yield id' *> sample


    sample' :: Params -> L.Context -> Int -> TVar [L.Token] -> IO L.Token
    sample' params' ctx nVocab lastNTokensTV = do
      alloca $ \candidatesPPtr ->
        allocaArray nVocab $ \logitsCopyPtr -> do

          -- logits are a multidimensional array:
          -- logits[_vocab][n_tokens], as best as I can tell from how
          -- it's used ("rows/cols" as described in the docstring
          -- seems backwards)
          logitsPtr <-  L.getLogits ctx
          copyArray logitsCopyPtr logitsPtr nVocab
          logitsCopyFPtr <- newForeignPtr_ logitsCopyPtr

          let
            logitsCopy = V.unsafeFromForeignPtr0 logitsCopyFPtr nVocab

            candidates = [0..(nVocab - 1)] <&> \n ->
              L.TokenData (fromIntegral n) (V.unsafeIndex logitsCopy n) 0.0

          candidatesPtr <- newArray candidates

          let
            candidatesP =
              L.TokenDataArray candidatesPtr (fromIntegral nVocab) False

          poke candidatesPPtr $ candidatesP

          lastNTokens' <- readTVarIO lastNTokensTV

          -- penalties

          withArray lastNTokens' $ \(lastNTokensPtr :: Ptr L.Token) -> do
            let
              lastNTokensLen = fromIntegral . length $ lastNTokens'

            -- float nl_logit = logits[llama_token_nl()];
            _nlLogit <- V.unsafeIndex logitsCopy . fromIntegral <$> L.tokenNl

            --
            -- lastNTokensPtr should be a pointer just to the last
            -- set of tokens at the end of lastNTokens matching the
            -- count of lastNRepeat, right now it's the entire thing
            --
            L.sampleRepetitionPenalty
              ctx candidatesPPtr lastNTokensPtr lastNTokensLen (_repeatPenalty params')

            L.sampleFrequencyAndPresencePenalties
              ctx
              candidatesPPtr
              lastNTokensPtr
              lastNTokensLen
              (_alphaFrequency params')
              (_alphaPresence params')

            -- todo
            -- if (!penalize_nl) {
            --     logits[llama_token_nl()] = nl_logit;
            -- }
            -- insert nlLogit into logits at index tokenNl

            -- sampling

            -- todo: other sampling methods
            -- id' <- L.sampleTokenGreedy ctx candidatesPPtr

            L.sampleTopK ctx candidatesPPtr (_topK params') 1
            L.sampleTailFree ctx candidatesPPtr (_tfsZ params') 1
            L.sampleTypical ctx candidatesPPtr (_typicalP params') 1
            L.sampleTopP ctx candidatesPPtr (_topP params') 1
            L.sampleTemperature ctx candidatesPPtr (_temp params')
            L.sampleToken ctx candidatesPPtr


    initialize :: IO (Params, Ptr L.ContextParams, L.Context, L.Model)
    initialize = do
      let
        opts = info (paramsParser <**> helper)
          ( fullDesc
         <> progDesc "run a single prompt (for now) against a model"
          )

      params' <- execParser opts

      putStrLn "\ninitBackend"
      L.initBackend (_enableNumaOpts params')

      -- todo
      -- putStrLn "\ndefault model quantize params"

      cpp <- malloc
      putStrLn "\ninit context params"
      L.contextDefaultParams cpp
      ctxParams' <- peek cpp

      let
        ctxParams = ctxParams'
          { L._nCtx = _nCtx params'
          , L._nGpuLayers = _nGpuLayers params'
          }

      poke cpp ctxParams

      putStrLn "\nloading model"

      model' <- withCString (_modelPath params') $ flip L.loadModelFromFile cpp

      putStrLn "\nloading context"

      ctx <- L.newContextWithModel model' cpp
      pure (params', cpp, ctx, model')


    cleanup :: (Params, Ptr L.ContextParams, L.Context, L.Model) -> IO ()
    cleanup (_params, cpp, ctx, model') = do
      putStrLn "\n\nfreeing context, model, context params"

      L.printTimings ctx

      L.free ctx
      L.freeModel model'
      free cpp

      L.freeBackend

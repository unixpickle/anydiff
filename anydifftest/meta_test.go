package anydifftest

import (
	"testing"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anyvec"
)

func TestPool(t *testing.T) {
	runWithCreators(t, func(t *testing.T, c anyvec.Creator, prec float64) {
		v := makeRandomVec(c, 18)
		ch := &ResChecker{
			F: func() anydiff.Res {
				return anydiff.Pool(anydiff.Tanh(v), func(r anydiff.Res) anydiff.Res {
					return anydiff.Mul(r, r)
				})
			},
			V: []*anydiff.Var{v},
		}
		ch.FullCheck(t)
	})
}

func TestPoolMulti(t *testing.T) {
	runWithCreators(t, func(t *testing.T, c anyvec.Creator, prec float64) {
		v1 := makeRandomVec(c, 18)
		v2 := makeRandomVec(c, 18)
		ch := &ResChecker{
			F: func() anydiff.Res {
				mIn := anydiff.Fuse(v1, v2)
				mOut := anydiff.PoolMulti(mIn, func(reses []anydiff.Res) anydiff.MultiRes {
					res1 := anydiff.Sub(reses[0], reses[1])
					res2 := anydiff.Sub(reses[1], anydiff.Mul(reses[0], reses[1]))
					return anydiff.Fuse(res1, res2)
				})
				return anydiff.Unfuse(mOut, func(reses []anydiff.Res) anydiff.Res {
					return anydiff.Sub(reses[0], reses[1])
				})
			},
			V: []*anydiff.Var{v1, v2},
		}
		ch.FullCheck(t)
	})
}

func TestSeqPool(t *testing.T) {
	runWithCreators(t, func(t *testing.T, c anyvec.Creator, prec float64) {
		inSeq, varList := makeBasicTestSeqs(c)
		ch := &SeqChecker{
			F: func() anyseq.Seq {
				return anyseq.Pool(inSeq, func(s anyseq.Seq) anyseq.Seq {
					return anyseq.Map(s, func(v anydiff.Res, n int) anydiff.Res {
						return anydiff.Tanh(v)
					})
				})
			},
			V: varList,
		}
		ch.FullCheck(t)
	})
}

func TestSeqPoolAsym(t *testing.T) {
	runWithCreators(t, func(t *testing.T, c anyvec.Creator, prec float64) {
		inSeq, varList := makeBasicTestSeqs(c)
		outSeq := anyseq.ConstSeqList(c, [][]anyvec.Vector{
			{c.MakeVectorData(c.MakeNumericList([]float64{1, 2}))},
		})
		ch := &SeqChecker{
			F: func() anyseq.Seq {
				return anyseq.Pool(inSeq, func(s anyseq.Seq) anyseq.Seq {
					return outSeq
				})
			},
			V: varList,
		}
		ch.FullCheck(t)
	})
}

func TestSeqPoolToVec(t *testing.T) {
	runWithCreators(t, func(t *testing.T, c anyvec.Creator, prec float64) {
		inSeq, varList := makeBasicTestSeqs(c)
		ch := &ResChecker{
			F: func() anydiff.Res {
				return anyseq.PoolToVec(inSeq, func(s anyseq.Seq) anydiff.Res {
					return anyseq.Sum(s)
				})
			},
			V: varList,
		}
		ch.FullCheck(t)
	})
}

func TestSeqPoolFromVec(t *testing.T) {
	runWithCreators(t, func(t *testing.T, c anyvec.Creator, prec float64) {
		inSeq, varList := makeBasicTestSeqs(c)
		inVar := anydiff.NewVar(c.MakeVector(1))
		inVar.Vector.AddScalar(c.MakeNumeric(0.5))
		ch := &SeqChecker{
			F: func() anyseq.Seq {
				squashed := anydiff.Tanh(inVar)
				return anyseq.PoolFromVec(squashed, func(r anydiff.Res) anyseq.Seq {
					return anyseq.Map(inSeq, func(v anydiff.Res, n int) anydiff.Res {
						return anydiff.ScaleRepeated(v, r)
					})
				})
			},
			V: append([]*anydiff.Var{inVar}, varList...),
		}
		ch.FullCheck(t)
	})
}

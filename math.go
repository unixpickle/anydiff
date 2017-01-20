package anydiff

import "github.com/unixpickle/anyvec"

type tanhRes struct {
	In     Res
	OutVec anyvec.Vector
}

// Tanh computes the hyperbolic tangent of each component
// of the input.
func Tanh(in Res) Res {
	v := in.Output().Copy()
	anyvec.Tanh(v)
	return &tanhRes{
		In:     in,
		OutVec: v,
	}
}

func (t *tanhRes) Output() anyvec.Vector {
	return t.OutVec
}

func (t *tanhRes) Vars() VarSet {
	return t.In.Vars()
}

func (t *tanhRes) Propagate(u anyvec.Vector, g Grad) {
	down := t.OutVec.Copy()
	anyvec.Pow(down, t.OutVec.Creator().MakeNumeric(2))
	anyvec.Complement(down)
	u.Mul(down)
	t.In.Propagate(u, g)
}

type logSoftmaxRes struct {
	In        Res
	ChunkSize int
	OutVec    anyvec.Vector
}

// LogSoftmax computes the log of the softmax function for
// each chunk in a packed list of chunks.
// The chunk size must divide the vector length.
// If chunkSize is 0, it will be treated like the full
// length of v.
func LogSoftmax(v Res, chunkSize int) Res {
	if chunkSize == 0 {
		chunkSize = v.Output().Len()
	}
	if v.Output().Len()%chunkSize != 0 {
		panic("chunk size must divide vector size")
	}
	out := v.Output().Copy()
	anyvec.LogSoftmax(out, chunkSize)
	return &logSoftmaxRes{
		In:        v,
		ChunkSize: chunkSize,
		OutVec:    out,
	}
}

func (l *logSoftmaxRes) Output() anyvec.Vector {
	return l.OutVec
}

func (l *logSoftmaxRes) Vars() VarSet {
	return l.In.Vars()
}

func (l *logSoftmaxRes) Propagate(u anyvec.Vector, g Grad) {
	numBatch := u.Len() / l.ChunkSize

	batchSums := anyvec.SumCols(u, numBatch)
	probs := l.OutVec.Copy()
	anyvec.Exp(probs)
	anyvec.ScaleChunks(probs, batchSums)
	u.Sub(probs)

	l.In.Propagate(u, g)
}

// Square squares the vector components.
func Square(v Res) Res {
	return Pool(v, func(v Res) Res {
		return Mul(v, v)
	})
}

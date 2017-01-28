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

type sigmoidRes struct {
	In     Res
	OutVec anyvec.Vector
}

// Sigmoid computes the logistic sigmoid of the input.
//
// The sigmoid is defined as:
//
//     f(x) = 1 / (1 + exp(-x))
//
func Sigmoid(in Res) Res {
	res := in.Output().Copy()
	anyvec.Sigmoid(res)
	return &sigmoidRes{
		In:     in,
		OutVec: res,
	}
}

func (s *sigmoidRes) Output() anyvec.Vector {
	return s.OutVec
}

func (s *sigmoidRes) Vars() VarSet {
	return s.In.Vars()
}

func (s *sigmoidRes) Propagate(u anyvec.Vector, g Grad) {
	comp := s.OutVec.Copy()
	u.Mul(comp)
	anyvec.Complement(comp)
	u.Mul(comp)
	s.In.Propagate(u, g)
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
	return Pow(v, v.Output().Creator().MakeNumeric(2))
}

type powRes struct {
	In     Res
	OutVec anyvec.Vector
	Power  anyvec.Numeric
}

// Pow raises each component of the vector to the given
// scaler power.
func Pow(v Res, s anyvec.Numeric) Res {
	out := v.Output().Copy()
	anyvec.Pow(out, s)
	return &powRes{
		In:     v,
		OutVec: out,
		Power:  s,
	}
}

func (p *powRes) Output() anyvec.Vector {
	return p.OutVec
}

func (p *powRes) Vars() VarSet {
	return p.In.Vars()
}

func (p *powRes) Propagate(u anyvec.Vector, g Grad) {
	temp := u.Creator().MakeVector(1)
	temp.AddScaler(p.Power)
	temp.AddScaler(temp.Creator().MakeNumeric(-1))
	powerMinusOne := anyvec.Sum(temp)

	exped := p.In.Output().Copy()
	anyvec.Pow(exped, powerMinusOne)
	u.Mul(exped)
	u.Scale(p.Power)

	p.In.Propagate(u, g)
}

type clipPosRes struct {
	In     Res
	OutVec anyvec.Vector
}

// ClipPos clips the values to be non-negative.
// Thus, any negative entries of in are 0 in the result.
func ClipPos(in Res) Res {
	out := in.Output().Copy()
	anyvec.ClipPos(out)
	return &clipPosRes{
		In:     in,
		OutVec: out,
	}
}

func (c *clipPosRes) Output() anyvec.Vector {
	return c.OutVec
}

func (c *clipPosRes) Vars() VarSet {
	return c.In.Vars()
}

func (c *clipPosRes) Propagate(u anyvec.Vector, g Grad) {
	mask := c.In.Output().Copy()
	anyvec.GreaterThan(mask, mask.Creator().MakeNumeric(0))
	u.Mul(mask)
	c.In.Propagate(u, g)
}

type sinRes struct {
	In     Res
	OutVec anyvec.Vector
}

// Sin takes the component-wise sine of the vector, in
// radians.
func Sin(in Res) Res {
	out := in.Output().Copy()
	anyvec.Sin(out)
	return &sinRes{
		In:     in,
		OutVec: out,
	}
}

func (s *sinRes) Output() anyvec.Vector {
	return s.OutVec
}

func (s *sinRes) Vars() VarSet {
	return s.In.Vars()
}

func (s *sinRes) Propagate(u anyvec.Vector, g Grad) {
	upScale := s.In.Output().Copy()
	anyvec.Cos(upScale)
	u.Mul(upScale)
	s.In.Propagate(u, g)
}

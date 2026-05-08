/*
 * Copyright (c) 2014, Yawning Angel <yawning at schwanenlied dot me>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

// obfuscation protocol.
package moegen // import "PluggableTransport/transports/moegen"

import (
	"PluggableTransport/common/drbg"
	"PluggableTransport/common/log"
	"PluggableTransport/common/ntor"
	"PluggableTransport/common/probdist"
	"PluggableTransport/common/replayfilter"
	"PluggableTransport/transports/base"
	"PluggableTransport/transports/obfs4/framing"
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"

	"fmt"
	pt "git.torproject.org/pluggable-transports/goptlib.git"
	"io"
	"io/ioutil"
	"math/rand"
	"net"
	"sort"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
)

const (
	transportName = "moegen"

	nodeIDArg        = "node-id"
	publicKeyArg     = "public-key"
	privateKeyArg    = "private-key"
	seedArg          = "drbg-seed"
	certArg          = "cert"
	perturbationsArg = "perturbations" //format:[[1,2],[23,10],[44,12],[50,3]]

	// 	biasCmdArg = "moegen-distBias"

	seedLength             = drbg.SeedLength
	headerLength           = framing.FrameOverhead + packetOverhead
	clientHandshakeTimeout = time.Duration(60) * time.Second
	serverHandshakeTimeout = time.Duration(30) * time.Second
	replayTTL              = time.Duration(3) * time.Hour

	maxCloseDelay = 60
)

// biasedDist controls if the probability table will be ScrambleSuit style or
// uniformly distributed.
var biasedDist bool

type moegenClientArgs struct {
	nodeID          *ntor.NodeID    //useless
	publicKey       *ntor.PublicKey //useless
	sessionKey      *ntor.Keypair
	dualPertManager *DualPerturbationManager // perturbations
}

// Transport is the moegen implementation of the base.Transport interface.
type Transport struct{}

type Perturbation struct {
	Index int
	Count int
}

type PerturbationManager struct {
	PerturbationList []Perturbation // PerturbationList: raising order list
	Pointer          int            // Pointer: current point value
}

func (pm *PerturbationManager) PMdeepcopy() *PerturbationManager {
	list := make([]Perturbation, len(pm.PerturbationList))
	copy(list, pm.PerturbationList)
	return &PerturbationManager{list, 0}
}

type DualPerturbationManager struct {
	Odd  *PerturbationManager // Client-->Bridge burst, note idx has been transferred, e.g., origin idx 5 --> idx 3
	Even *PerturbationManager // Bridge-->Client burst, origin idx 6 --> idx 3
}

func (dpm *DualPerturbationManager) deepcopy() *DualPerturbationManager {
	if dpm == nil {
		return nil
	}
	return &DualPerturbationManager{
		Odd:  dpm.Odd.PMdeepcopy(),
		Even: dpm.Even.PMdeepcopy(),
	}
}

func ParseDualPerturbations(content string) (*DualPerturbationManager, error) {
	// 1）JSON parse
	var raw [][]int
	if err := json.Unmarshal([]byte(content), &raw); err != nil {
		return nil, errors.New("parse failure: expected [[idx,count],...] format")
	}

	// 2）split odd/even into map
	oddMap := make(map[int]int)
	evenMap := make(map[int]int)
	for _, pair := range raw {
		if len(pair) != 2 {
			log.Noticef("-----------[moegen.ParseDualPerturbations()] each element must be a [index,count] pair, got: %v", pair)
			fmt.Println(fmt.Errorf("-----------[moegen.ParseDualPerturbations()] each element must be a [index,count] pair, got: %v", pair))
			return nil, errors.New("[moegen.ParseDualPerturbations()] each element must be a [index,count] pair")
		}
		idx, cnt := pair[0], pair[1]
		if idx%2 == 0 {
			evenMap[idx] += cnt
		} else {
			oddMap[idx] += cnt
		}
	}

	// 3）build []Perturbation from map
	buildManager := func(m map[int]int) *PerturbationManager {
		pList := make([]Perturbation, 0, len(m))
		for idx, cnt := range m {
			// note the idx should ceil(idx/2), e.g., odd's idx 3 is the corresponding idx 2 for burst client-->server.
			pList = append(pList, Perturbation{Index: (idx + 1) / 2, Count: cnt})
		}
		sort.Slice(pList, func(i, j int) bool {
			return pList[i].Index < pList[j].Index
		})
		return &PerturbationManager{PerturbationList: pList, Pointer: 0}
	}

	return &DualPerturbationManager{
		Odd:  buildManager(oddMap),
		Even: buildManager(evenMap),
	}, nil
}

// Name returns the name of the moegen transport protocol.
func (t *Transport) Name() string {
	return transportName
}

// ClientFactory returns a new moegenClientFactory instance.
func (t *Transport) ClientFactory(stateDir string) (base.ClientFactory, error) {
	cf := &moegenClientFactory{transport: t, stateDir: stateDir}
	log.Noticef("-----------[moegen.ClientFactory()] Already created ClientFactory")
	fmt.Println(fmt.Errorf("-----------[moegen.ClientFactory()] Already created ClientFactory"))
	return cf, nil
}

// ServerFactory returns a new moegenServerFactory instance.
func (t *Transport) ServerFactory(stateDir string, args *pt.Args) (base.ServerFactory, error) {
	st, err := serverStateFromArgs(stateDir, args)
	if err != nil {
		return nil, err
	}

	// Store the arguments that should appear in our descriptor for the clients.
	ptArgs := pt.Args{}
	ptArgs.Add(certArg, st.cert.String())               // if want save more args
	ptArgs.Add(perturbationsArg, st.PerturbationString) // Get PerturbationString (str list)

	cwd, _ := os.Getwd()
	log.Noticef("[[[[[--- Current working directory ---]]]]: %s", cwd)
	log.Noticef("[[[[[--- Current state directory ---]]]]: %s", stateDir)
	path := filepath.Join(stateDir, st.PerturbationString)
	content, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("-----------[moegen.ServerFactory()] failed to read perturbations file '%s': %v", path, err)
	}
	log.Noticef("-----------[moegen.ServerFactory()] Loaded perturbations from file: %s", path)

	dualPertManager, err := ParseDualPerturbations(string(content))
	if err != nil {
		log.Errorf("-----------[moegen.ServerFactory()] Error load perturbation list: %v", err)
		fmt.Println(fmt.Errorf("-----------[moegen.ServerFactory()] Error load perturbation list: %v", err))
	}
	log.Noticef("-----------[moegen.ServerFactory()] dualPertManager generated success.")
	fmt.Println(fmt.Errorf("-----------[moegen.ServerFactory()] dualPertManager generated success."))

	// Initialize the replay filter.
	filter, err := replayfilter.New(replayTTL)
	if err != nil {
		return nil, err
	}

	// Initialize the close thresholds for failed connections.
	drbg_, err := drbg.NewHashDrbg(st.drbgSeed)
	if err != nil {
		return nil, err
	}
	rng := rand.New(drbg_)

	sf := &moegenServerFactory{t, &ptArgs, st.nodeID, st.identityKey, st.drbgSeed,
		filter, rng.Intn(maxCloseDelay), dualPertManager}
	return sf, nil
}

type moegenClientFactory struct {
	transport       base.Transport
	dualPertManager *DualPerturbationManager
	stateDir        string
}

func (cf *moegenClientFactory) Transport() base.Transport {
	return cf.transport
}

func (cf *moegenClientFactory) ParseArgs(args *pt.Args) (interface{}, error) {
	var nodeID *ntor.NodeID
	var publicKey *ntor.PublicKey
	// The "new" (version >= 0.0.3) bridge lines use a unified "cert" argument
	// for the Node ID and Public Key.
	log.Noticef("-----------[moegen.ParseArgs()] beginning.")
	fmt.Println(fmt.Errorf("-----------[moegen.ParseArgs()] beginning."))
	certStr, ok := args.Get(certArg)
	if ok {
		cert, err := serverCertFromString(certStr)
		if err != nil {
			return nil, err
		}
		nodeID, publicKey = cert.unpack()
	} else {
		// The "old" style (version <= 0.0.2) bridge lines use separate Node ID
		// and Public Key arguments in Base16 encoding and are a UX disaster.
		nodeIDStr, ok := args.Get(nodeIDArg)
		if !ok {
			fmt.Println(fmt.Errorf("-----------[moegen.ParseArgs()] missing argument '%s'", nodeIDArg))
			return nil, fmt.Errorf("-----------[moegen.ParseArgs()] missing argument '%s'", nodeIDArg)
		}
		var err error
		if nodeID, err = ntor.NodeIDFromHex(nodeIDStr); err != nil {
			return nil, err
		}
		log.Noticef("-----------[moegen.ParseArgs()] nodeIDStr=%v", nodeIDStr)
		fmt.Println(fmt.Errorf("-----------[moegen.ParseArgs()] nodeIDStr=%v", nodeIDStr))

		publicKeyStr, ok := args.Get(publicKeyArg)
		if !ok {
			fmt.Println(fmt.Errorf("-----------[moegen.ParseArgs()] missing argument '%s'", publicKeyArg))
			return nil, fmt.Errorf("-----------[moegen.ParseArgs()] missing argument '%s'", publicKeyArg)
		}
		if publicKey, err = ntor.PublicKeyFromHex(publicKeyStr); err != nil {
			return nil, err
		}
	}

	pertPathStr, pertOk := args.Get(perturbationsArg)
	if !pertOk {
		fmt.Println(fmt.Errorf("-----------[moegen.ParseArgs()] missing argument '%s'", perturbationsArg))
		return nil, fmt.Errorf("-----------[moegen.ParseArgs()] missing argument '%s'", perturbationsArg)
	}
	cwd, _ := os.Getwd()
	log.Noticef("[[[[[--- Current working directory ---]]]]: %s", cwd)
	log.Noticef("[[[[[--- Current state directory ---]]]]: %s", cf.stateDir)
	path := filepath.Join(cf.stateDir, pertPathStr)
	content, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("-----------[moegen.ParseArgs()] failed to read perturbations file '%s': %v", path, err)
	}
	log.Noticef("-----------[moegen.ParseArgs()] Loaded perturbations from file: %s", path)
	dualPertManager, err := ParseDualPerturbations(string(content))

	if err != nil {
		log.Errorf("-----------[moegen.ParseArgs()] Error load perturbation list: %v", err)
		fmt.Println(fmt.Errorf("-----------[moegen.ParseArgs()] Error load perturbation list: %v", err))
	}
	cf.dualPertManager = dualPertManager // load perturbations by torrc perturbationsArg
	log.Noticef("-----------[moegen.ParseArgs()] dualPertManager generated success.")
	fmt.Println(fmt.Errorf("-----------[moegen.ParseArgs()] dualPertManager generated success."))

	// Generate the session key pair before connectiong to hide the Elligator2
	// rejection sampling from network observers.
	log.Noticef("-----------[moegen.ParseArgs()] Begin get sessionKey")
	fmt.Println(fmt.Errorf("-----------[moegen.ParseArgs()] Begin get sessionKey"))
	sessionKey, err := ntor.NewKeypair(true)
	if err != nil {
		return nil, err
	}

	log.Noticef("-----------[moegen.ParseArgs()] Already got sessionKey")
	fmt.Println(fmt.Errorf("-----------[moegen.ParseArgs()] Already got sessionKey"))
	return &moegenClientArgs{nodeID, publicKey, sessionKey, dualPertManager}, nil
}

func (cf *moegenClientFactory) Dial(network, addr string, dialFn base.DialFunc, args interface{}) (net.Conn, error) {
	// Validate args before bothering to open connection.
	log.Noticef("-----------[moegen.Dial()] Begin dial and get moegenClientArgs")
	fmt.Println(fmt.Errorf("-----------[moegen.Dial()] Begin dial and get moegenClientArgs"))
	ca, ok := args.(*moegenClientArgs)
	if !ok {
		return nil, fmt.Errorf("invalid argument type for args")
	}
	log.Noticef("-----------[moegen.Dial()] Already dial prepare get moegenClientConn")
	fmt.Println(fmt.Errorf("-----------[moegen.Dial()] Already dial prepare get moegenClientConn (ca=%v ok=%v)", ca, ok))
	conn, err := dialFn(network, addr)
	if err != nil {
		fmt.Println(fmt.Errorf("-----------[moegen.Dial()] dialFn error:%v, network=%v, addr=%v", err, network, addr))
		return nil, err
	}
	dialConn := conn
	if conn, err = newmoegenClientConn(conn, ca); err != nil {
		dialConn.Close()
		fmt.Println(fmt.Errorf("-----------[moegen.Dial()] create newmoegenClientConn error:%v", err))
		return nil, err
	}
	log.Noticef("-----------[moegen.Dial()] Already got moegenClientConn")
	fmt.Println(fmt.Errorf("-----------[moegen.Dial()] Already got moegenClientConn"))
	return conn, nil
}

type moegenServerFactory struct {
	transport base.Transport
	args      *pt.Args // pt: pluggable-transport goptlib pakage name

	nodeID       *ntor.NodeID
	identityKey  *ntor.Keypair
	lenSeed      *drbg.Seed
	replayFilter *replayfilter.ReplayFilter

	closeDelay      int
	dualPertManager *DualPerturbationManager
}

func (sf *moegenServerFactory) Transport() base.Transport {
	return sf.transport
}

func (sf *moegenServerFactory) Args() *pt.Args {
	return sf.args
}

func (sf *moegenServerFactory) WrapConn(conn net.Conn) (net.Conn, error) {
	// Not much point in having a separate newmoegenServerConn routine when
	// wrapping requires using values from the factory instance.

	// Generate the session keypair *before* consuming data from the peer, to
	// attempt to mask the rejection sampling due to use of Elligator2.  This
	// might be futile, but the timing differential isn't very large on modern
	// hardware, and there are far easier statistical attacks that can be
	// mounted as a distinguisher.
	log.Noticef("-----------[moegen.WrapConn()] Begin WrapConn and get sessionKey")
	fmt.Println(fmt.Errorf("-----------[moegen.WrapConn()] Begin WrapConn and get sessionKey"))
	sessionKey, err := ntor.NewKeypair(true)
	if err != nil {
		return nil, err
	}

	log.Noticef("-----------[moegen.WrapConn()] Already got WrapConn and create moegenConn")
	fmt.Println(fmt.Errorf("-----------[moegen.WrapConn()] Already got WrapConn and create moegenConn"))
	lenDist := probdist.New(sf.lenSeed, 0, framing.MaximumSegmentLength, biasedDist)
	loggerChan := make(chan []int64, 100)

	c := &moegenConn{conn, true, lenDist,
		bytes.NewBuffer(nil), bytes.NewBuffer(nil),
		make([]byte, consumeReadSize), nil, nil, loggerChan, 0, 0,
		sf.dualPertManager.deepcopy(), sync.Mutex{}, make(chan PacketInfo, 65535)}

	log.Noticef("-----------[moegen.WrapConn()] Already created moegenConn and begin serverHandshake")
	fmt.Println(fmt.Errorf("-----------[moegen.WrapConn()] Already created moegenConn and begin serverHandshake"))
	startTime := time.Now()

	if err = c.serverHandshake(sf, sessionKey); err != nil {
		c.closeAfterDelay(sf, startTime)
		return nil, err
	}
	log.Noticef("-----------[moegen.WrapConn()] Already ServerHandshake")
	fmt.Println(fmt.Errorf("-----------[moegen.WrapConn()] Already ServerHandshake"))

	return c, nil
}

type moegenConn struct {
	net.Conn

	isServer bool

	lenDist *probdist.WeightedDist

	receiveBuffer        *bytes.Buffer
	receiveDecodedBuffer *bytes.Buffer
	readBuffer           []byte

	encoder *framing.Encoder
	decoder *framing.Decoder

	loggerChan         chan []int64
	clientBurstCounter int64 // the amount of client sent burst
	serverBurstCounter int64 // the amount of server sent burst
	dualPertManager    *DualPerturbationManager

	mu       sync.Mutex
	sendChan chan PacketInfo // as struct element for packet.readPackets() using.
}

func newmoegenClientConn(conn net.Conn, args *moegenClientArgs) (c *moegenConn, err error) {
	// Generate the initial protocol polymorphism distribution(s).
	var seed *drbg.Seed
	log.Noticef("-----------[moegen.newmoegenClientConn()] Begin get moegenConn parameters")
	fmt.Println(fmt.Errorf("-----------[moegen.newmoegenClientConn()] Begin get moegenConn parameters"))
	if seed, err = drbg.NewSeed(); err != nil {
		return
	}
	lenDist := probdist.New(seed, 0, framing.MaximumSegmentLength, biasedDist)
	loggerChan := make(chan []int64, 100)

	log.Noticef("-----------[moegen.newmoegenClientConn()] Begin create moegenConn")
	fmt.Println(fmt.Errorf("-----------[moegen.newmoegenClientConn()] Begin create moegenConn"))
	// Allocate the client structure.
	c = &moegenConn{conn, false, lenDist,
		bytes.NewBuffer(nil), bytes.NewBuffer(nil),
		make([]byte, consumeReadSize), nil, nil, loggerChan, 0, 0,
		args.dualPertManager.deepcopy(), sync.Mutex{}, make(chan PacketInfo, 65535)}

	// Start the handshake timeout.
	deadline := time.Now().Add(clientHandshakeTimeout)
	if err = conn.SetDeadline(deadline); err != nil {
		return nil, err
	}

	log.Noticef("-----------[moegen.newmoegenClientConn()] Begin clientHandshake")
	fmt.Println(fmt.Errorf("-----------[moegen.newmoegenClientConn()] Begin clientHandshake"))
	if err = c.clientHandshake(args.nodeID, args.publicKey, args.sessionKey); err != nil {
		return nil, err
	}

	// Stop the handshake timeout.
	if err = conn.SetDeadline(time.Time{}); err != nil {
		return nil, err
	}

	log.Noticef("-----------[moegen.newmoegenClientConn()] Already clientHandshake")
	fmt.Println(fmt.Errorf("-----------[moegen.newmoegenClientConn()] Already clientHandshake"))
	return
}

func (conn *moegenConn) clientHandshake(nodeID *ntor.NodeID, peerIdentityKey *ntor.PublicKey, sessionKey *ntor.Keypair) error {
	log.Noticef("-----------[moegen.clientHandshake()] Begin clientHandshake")
	fmt.Println(fmt.Errorf("-----------[moegen.clientHandshake()] Begin clientHandshake"))
	if conn.isServer {
		return fmt.Errorf("clientHandshake called on server connection")
	}

	// Generate and send the client handshake.
	hs := newClientHandshake(nodeID, peerIdentityKey, sessionKey)
	blob, err := hs.generateHandshake()
	if err != nil {
		return err
	}
	if _, err = conn.Conn.Write(blob); err != nil {
		return err
	}
	nextIdx := atomic.AddInt64(&conn.clientBurstCounter, 1) // handshake request also is burst
	conn.CheckAndSendDummyCell(nextIdx)                     // request may also insert dummy

	//log.Noticef("-----------[moegen.clientHandshake()] Already create clientHandshake")
	//fmt.Println(fmt.Errorf("-----------[moegen.clientHandshake()] Already create clientHandshake"))
	// Consume the server handshake.
	var hsBuf [maxHandshakeLength]byte
	for {
		//log.Noticef("-----------[moegen.clientHandshake()] For looping Client Handshake")
		//fmt.Println(fmt.Errorf("-----------[moegen.clientHandshake()] For looping Client Handshake"))
		n, err := conn.Conn.Read(hsBuf[:])
		if err != nil {
			// The Read() could have returned data and an error, but there is
			// no point in continuing on an EOF or whatever.
			log.Noticef("-----------[moegen.clientHandshake()] conn.Conn.Read error %v", err)
			fmt.Println(fmt.Errorf("-----------[moegen.clientHandshake()] conn.Conn.Read error %v", err))
			return err
		}
		conn.receiveBuffer.Write(hsBuf[:n])

		//log.Noticef("-----------[moegen.clientHandshake()] begin parse ServerHandshake")
		//fmt.Println(fmt.Errorf("-----------[moegen.clientHandshake()] begin parse ServerHandshake"))
		n, seed, err := hs.parseServerHandshake(conn.receiveBuffer.Bytes())
		if err == ErrMarkNotFoundYet {
			continue
		} else if err != nil {
			log.Noticef("-----------[moegen.clientHandshake()] ServerHandshake args parse error: %v", err)
			fmt.Println(fmt.Errorf("-----------[moegen.clientHandshake()] ServerHandshake args parse error: %v", err))
			return err
		}
		_ = conn.receiveBuffer.Next(n)

		//log.Noticef("-----------[moegen.clientHandshake()] begin intialize the link crypto")
		//fmt.Println(fmt.Errorf("-----------[moegen.clientHandshake()] begin intialize the link crypto"))
		// Use the derived key material to intialize the link crypto.
		okm := ntor.Kdf(seed, framing.KeyLength*2)
		conn.encoder = framing.NewEncoder(okm[:framing.KeyLength])
		conn.decoder = framing.NewDecoder(okm[framing.KeyLength:])

		log.Noticef("-----------[moegen.clientHandshake()] clientHandshake Done")
		fmt.Println(fmt.Errorf("-----------[moegen.clientHandshake()] clientHandshake Done"))
		return nil
	}
}

func (conn *moegenConn) serverHandshake(sf *moegenServerFactory, sessionKey *ntor.Keypair) error {
	log.Noticef("-----------[moegen.serverHandshake()] Begin serverHandshake")
	fmt.Println(fmt.Errorf("-----------[moegen.serverHandshake()] Begin serverHandshake"))
	if !conn.isServer {
		return fmt.Errorf("serverHandshake called on client connection")
	}

	// Generate the server handshake, and arm the base timeout.
	hs := newServerHandshake(sf.nodeID, sf.identityKey, sessionKey)
	if err := conn.Conn.SetDeadline(time.Now().Add(serverHandshakeTimeout)); err != nil {
		return err
	}

	//log.Noticef("-----------[moegen.serverHandshake()] Handshake created")
	//fmt.Println(fmt.Errorf("-----------[moegen.serverHandshake()] Handshake created"))
	// Consume the client handshake.
	var hsBuf [maxHandshakeLength]byte
	for {
		//log.Noticef("-----------[moegen.serverHandshake()] For looping Server Handshake")
		//fmt.Println(fmt.Errorf("-----------[moegen.serverHandshake()] For looping Server Handshake"))
		n, err := conn.Conn.Read(hsBuf[:])
		if err != nil {
			// The Read() could have returned data and an error, but there is
			// no point in continuing on an EOF or whatever.
			log.Noticef("-----------[moegen.serverHandshake()] Read failed error: %v", err)
			fmt.Println(fmt.Errorf("-----------[moegen.serverHandshake()] Read failed error: %v", err))
			return err
		}
		conn.receiveBuffer.Write(hsBuf[:n])

		seed, err := hs.parseClientHandshake(sf.replayFilter, conn.receiveBuffer.Bytes())
		if err == ErrMarkNotFoundYet {
			continue
		} else if err != nil {
			log.Noticef("-----------[moegen.serverHandshake()] ClientHandshake args parse failed error: %v", err)
			fmt.Println(fmt.Errorf("-----------[moegen.serverHandshake()] ClientHandshake args parse failed error: %v", err))
			return err
		}
		conn.receiveBuffer.Reset()

		if err := conn.Conn.SetDeadline(time.Time{}); err != nil {
			return nil
		}

		// Use the derived key material to intialize the link crypto.
		//log.Noticef("-----------[moegen.serverHandshake()] Begin intialize the link crypto")
		//fmt.Println(fmt.Errorf("-----------[moegen.serverHandshake()] Begin intialize the link crypt"))
		okm := ntor.Kdf(seed, framing.KeyLength*2)
		conn.encoder = framing.NewEncoder(okm[framing.KeyLength:])
		conn.decoder = framing.NewDecoder(okm[:framing.KeyLength])

		break
	}

	// Since the current and only implementation always sends a PRNG seed for
	// the length obfuscation, this makes the amount of data received from the
	// server inconsistent with the length sent from the client.
	//
	// Rebalance this by tweaking the client mimimum padding/server maximum
	// padding, and sending the PRNG seed unpadded (As in, treat the PRNG seed
	// as part of the server response).  See inlineSeedFrameLength in
	// handshake_ntor.go.

	// Generate/send the response.
	blob, err := hs.generateHandshake()
	if err != nil {
		return err
	}
	var frameBuf bytes.Buffer
	if _, err = frameBuf.Write(blob); err != nil {
		return err
	}

	// Send the PRNG seed as the first packet.
	if err := conn.makePacket(&frameBuf, packetTypePrngSeed, sf.lenSeed.Bytes()[:], 0); err != nil {
		return err
	}
	if _, err = conn.Conn.Write(frameBuf.Bytes()); err != nil {
		return err
	}
	nextIdx := atomic.AddInt64(&conn.serverBurstCounter, 1) // handshake response also is burst
	conn.CheckAndSendDummyCell(nextIdx)                     // response may also insert dummy

	log.Noticef("-----------[moegen.serverHandshake()] Send a cell with [pktType=%v, idx=%v] "+
		"(0=packetTypePayload  1=packetTypeDummy  2=packetTypePrngSeed)", packetTypePrngSeed, conn.serverBurstCounter) // note the idx may larger than true idx, cause rountine
	fmt.Println(fmt.Errorf("-----------[moegen.serverHandshake()] Send a cell with [pktType=%v, idx=%v] "+
		"(0=packetTypePayload  1=packetTypeDummy  2=packetTypePrngSeed)", packetTypePrngSeed, conn.serverBurstCounter))

	log.Noticef("-----------[moegen.serverHandshake()] Server Handshake Done")
	fmt.Println(fmt.Errorf("-----------[moegen.serverHandshake()] Server Handshake Done"))
	return nil
}

func (conn *moegenConn) Read(b []byte) (n int, err error) {
	// If there is no payload from the previous Read() calls, consume data off
	// the network.  Not all data received is guaranteed to be usable payload,
	// so do this in a loop till data is present or an error occurs.
	//log.Noticef("-----------[moegen.Read()] Begin Read Cell")
	//fmt.Println(fmt.Errorf("-----------[moegen.Read()] Begin Read Cell"))
	for conn.receiveDecodedBuffer.Len() == 0 {
		//log.Noticef("-----------[moegen.Read()] Receive empty buffer")
		//fmt.Println(fmt.Errorf("-----------[moegen.Read()] Receive empty buffer"))
		err = conn.readPackets()
		if err == framing.ErrAgain {
			// Don't proagate this back up the call stack if we happen to break
			// out of the loop.
			err = nil
			continue
		} else if err != nil {
			break
		}
	}

	// Even if err is set, attempt to do the read anyway so that all decoded
	// data gets relayed before the connection is torn down.
	if conn.receiveDecodedBuffer.Len() > 0 {
		//log.Noticef("-----------[moegen.Read()] Receive buffer with len=%v", conn.receiveDecodedBuffer.Len())
		//fmt.Println(fmt.Errorf("-----------[moegen.Read()] Receive buffer with len=%v", conn.receiveDecodedBuffer.Len()))
		var berr error
		n, berr = conn.receiveDecodedBuffer.Read(b)
		if err == nil {
			// Only propagate berr if there are not more important (fatal)
			// errors from the network/crypto/packet processing.
			err = berr
		}
	}

	return
}

func isDummyInsertIndex(nextIdx int64, pm *PerturbationManager) (bool, int) {
	if pm.Pointer < len(pm.PerturbationList) {
		desired := int64(pm.PerturbationList[pm.Pointer].Index)
		if nextIdx == desired {
			return true, pm.PerturbationList[pm.Pointer].Count
		} else if nextIdx > desired {
			log.Errorf("-----------[moegen.isDummyInsertIndex()] Occur next_burst_idx > next_dummy_idx:%v > %v", nextIdx, desired)
			fmt.Println(fmt.Errorf("-----------[moegen.isDummyInsertIndex()] Occur next_burst_idx > next_dummy_idx:%v > %v", nextIdx, desired))
			pm.Pointer++
		}
	}
	return false, 0
}

func (conn *moegenConn) CheckAndSendDummyCell(nextIdx int64) {
	currPertManager := conn.dualPertManager.Odd // Odd is client burst
	if conn.isServer {
		currPertManager = conn.dualPertManager.Even
	}

	flag, count := isDummyInsertIndex(nextIdx, currPertManager)
	if flag {
		// send goroutine call for dummy cell
		conn.mu.Lock()
		for i := 0; i < count; i++ {
			conn.sendChan <- PacketInfo{PktType: packetTypeDummy, Data: []byte{}, PadLen: maxPacketPaddingLength}
		}
		defer conn.mu.Unlock()
		// move to next perturbation
		if conn.isServer {
			conn.dualPertManager.Even.Pointer++
		} else {
			conn.dualPertManager.Odd.Pointer++
		}
	}
}

func (conn *moegenConn) ReadFrom(r io.Reader) (written int64, err error) {
	log.Noticef("-----------[State] Enter copyloop state: isServer=%d  & clientCounter=%d & serverCounter=%d",
		conn.isServer, conn.clientBurstCounter, conn.serverBurstCounter)
	fmt.Println(fmt.Errorf("-----------[State] Enter copyloop state: isServer=%d  & clientCounter=%d & serverCounter=%d",
		conn.isServer, conn.clientBurstCounter, conn.serverBurstCounter))
	closeChan := make(chan int)
	defer close(closeChan)
	errChan := make(chan error, 5)
	// sendChan  // it's define in construct moegenConn := make(chan PacketInfo, 65535)
	var receiveBuf bytes.Buffer //read payload from upstream and buffer here

	// 0) log goroutine: can be written

	// 1) send goroutine: Get PacketInfo from sendChan, and send to bridge other side
	go func() {
		for {
			select {
			case <-closeChan:
				return
			case packetInfo := <-conn.sendChan:
				pktType := packetInfo.PktType
				Data := packetInfo.Data
				PadLen := packetInfo.PadLen

				var frameBuf bytes.Buffer
				err = conn.makePacket(&frameBuf, pktType, Data, PadLen)
				if err != nil { // check generate packet
					errChan <- err
					return
				}
				_, wtErr := conn.Conn.Write(frameBuf.Bytes())
				idx := atomic.LoadInt64(&conn.clientBurstCounter)
				if conn.isServer {
					idx = atomic.LoadInt64(&conn.serverBurstCounter)
				}
				log.Noticef("-----------[moegen.ReadFrom() SendRoutine] Send a cell with [pktType=%v, burst=%v, isServer=%v] "+
					"(0=packetTypePayload  1=packetTypeDummy  2=packetTypePrngSeed)", pktType, idx, conn.isServer) // note the idx may larger than true idx, cause rountine
				fmt.Println(fmt.Errorf("-----------[moegen.ReadFrom() SendRoutine] Send a cell with [pktType=%v, burst=%v, isServer=%v] "+
					"(0=packetTypePayload  1=packetTypeDummy  2=packetTypePrngSeed)", pktType, idx, conn.isServer))

				if wtErr != nil {
					errChan <- wtErr
					log.Errorf("-----------[moegen.ReadFrom() SendRoutine] Send routine exits by write err.")
					fmt.Println(fmt.Errorf("-----------[moegen.ReadFrom() Routine] Send routine exits by write err."))
					return
				}
			}
		}
	}()

	// 2) main & dummy goroutine
	for {
		select {
		case e := <-errChan:
			// Background goroutine error, exit.
			log.Errorf("-----------[moegen.ReadFrom() MainRoutine] exiting due to error: %v", e)
			fmt.Println(fmt.Errorf("-----------[moegen.ReadFrom() MainRoutine] exiting due to error: %v", e))
			return written, e
		default:
			// Read upstream data
			buf := make([]byte, 65535)
			rdLen, rerr := r.Read(buf)
			log.Noticef("-----------[moegen.ReadFrom() MainRoutine] New burst, get %v length data for send.", rdLen)
			fmt.Println(fmt.Errorf("-----------[moegen.ReadFrom() MainRoutine] New burst, get %v length data for send.", rdLen))
			if rerr != nil {
				if rerr == io.EOF || rerr == io.ErrUnexpectedEOF {
					close(closeChan)
					return written, nil
				}
				return written, rerr
			}
			if rdLen == 0 {
				log.Errorf("----------- [moegen.ReadFrom() MainRoutine]BUG? read 0 bytes, err: %v", err)
				fmt.Println(fmt.Errorf("----------- [moegen.ReadFrom() MainRoutine]BUG? read 0 bytes, err: %v", err))
				continue
			}

			receiveBuf.Write(buf[:rdLen]) //from buf to receiveBuf
			// Cumulative writing
			written += int64(rdLen)
			// Slice by maximum frame length, send to sendChan
			for receiveBuf.Len() > 0 {
				chunk := make([]byte, maxPacketPayloadLength)
				m, rdErr := receiveBuf.Read(chunk)
				if rdErr != nil {
					log.Errorf("----------- [moegen.ReadFrom() MainRoutine] loop write err: %v", rdErr)
					fmt.Println(fmt.Errorf("----------- [moegen.ReadFrom() MainRoutine] loop write err: %v", rdErr))
					return written, rdErr
				}
				conn.sendChan <- PacketInfo{
					PktType: packetTypePayload,
					Data:    chunk[:m],
					PadLen:  maxPacketPaddingLength - uint16(m),
				}
			}
			var nextIdx int64
			if conn.isServer {
				nextIdx = atomic.AddInt64(&conn.serverBurstCounter, 1) // initial counter = 0, add --> idx 1
			} else {
				nextIdx = atomic.AddInt64(&conn.clientBurstCounter, 1)
			}
			// send dummy
			conn.CheckAndSendDummyCell(nextIdx)
		}
	}
}

func (conn *moegenConn) SetDeadline(t time.Time) error {
	return syscall.ENOTSUP
}

func (conn *moegenConn) SetWriteDeadline(t time.Time) error {
	return syscall.ENOTSUP
}

func (conn *moegenConn) closeAfterDelay(sf *moegenServerFactory, startTime time.Time) {
	// I-it's not like I w-wanna handshake with you or anything.  B-b-baka!
	log.Noticef("-----------[moegen.closeAfterDelay()] Prepare close for delay.")
	fmt.Println(fmt.Errorf("-----------[moegen.closeAfterDelay()] Prepare close for delay."))
	defer conn.Conn.Close()

	delay := time.Duration(sf.closeDelay)*time.Second + serverHandshakeTimeout
	deadline := startTime.Add(delay)
	if time.Now().After(deadline) {
		return
	}

	if err := conn.Conn.SetReadDeadline(deadline); err != nil {
		return
	}

	// Consume and discard data on this connection until the specified interval
	// passes.
	log.Noticef("-----------[moegen.closeAfterDelay()] Timeout, close.")
	fmt.Println(fmt.Errorf("-----------[moegen.closeAfterDelay()] Timeout, close."))
	_, _ = io.Copy(ioutil.Discard, conn.Conn)
}

var _ base.ClientFactory = (*moegenClientFactory)(nil)
var _ base.ServerFactory = (*moegenServerFactory)(nil)
var _ base.Transport = (*Transport)(nil)
var _ net.Conn = (*moegenConn)(nil)

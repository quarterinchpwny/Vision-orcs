import { Module } from "../../compiler/parser/Module.js";
import { Klass } from "../../compiler/types/Class.js";
import { doublePrimitiveType, stringPrimitiveType } from "../../compiler/types/PrimitiveTypes.js";
import { Method, Parameterlist } from "../../compiler/types/Types.js";
import { RuntimeObject } from "../../interpreter/RuntimeObject.js";
import { FilledShapeHelper } from "./FilledShape.js";
import { WorldHelper } from "./World.js";
import { EnumRuntimeObject } from "../../compiler/types/Enum.js";
import { Interpreter } from "../../interpreter/Interpreter.js";

export class TextClass extends Klass {

    constructor(module: Module) {

        super("Text", module, "Text, der innerhalb der Grafikausgabe dargestellt werden kann");

        this.setBaseClass(<Klass>module.typeStore.getType("FilledShape"));

        // this.addAttribute(new Attribute("PI", doublePrimitiveType, (object) => { return Math.PI }, true, Visibility.public, true, "Die Kreiszahl Pi (3.1415...)"));

        this.addMethod(new Method("Text", new Parameterlist([
            { identifier: "x", type: doublePrimitiveType, declaration: null, usagePositions: null, isFinal: true },
            { identifier: "y", type: doublePrimitiveType, declaration: null, usagePositions: null, isFinal: true },
            { identifier: "fontsize", type: doublePrimitiveType, declaration: null, usagePositions: null, isFinal: true },
            { identifier: "text", type: stringPrimitiveType, declaration: null, usagePositions: null, isFinal: true },
        ]), null,
            (parameters) => {

                let o: RuntimeObject = parameters[0].value;
                let x: number = parameters[1].value;
                let y: number = parameters[2].value;
                let fontsize: number = parameters[3].value;
                let text: string = parameters[4].value;

                let sh = new TextHelper(x, y, fontsize, text, module.main.getInterpreter(), o);
                o.intrinsicData["Actor"] = sh;

            }, false, false, 'Instanziert ein neues Textobjekt. (x, y) sind die Koordinaten des Textankers (default: links oben), fontsize die Höhe des Textes in Pixeln.', true));

        this.addMethod(new Method("Text", new Parameterlist([
            { identifier: "x", type: doublePrimitiveType, declaration: null, usagePositions: null, isFinal: true },
            { identifier: "y", type: doublePrimitiveType, declaration: null, usagePositions: null, isFinal: true },
            { identifier: "fontsize", type: doublePrimitiveType, declaration: null, usagePositions: null, isFinal: true },
            { identifier: "text", type: stringPrimitiveType, declaration: null, usagePositions: null, isFinal: true },
            { identifier: "font-family", type: stringPrimitiveType, declaration: null, usagePositions: null, isFinal: true }
        ]), null,
            (parameters) => {

                let o: RuntimeObject = parameters[0].value;
                let x: number = parameters[1].value;
                let y: number = parameters[2].value;
                let fontsize: number = parameters[3].value;
                let text: string = parameters[4].value;
                let fontFamily: string = parameters[5].value;

                let sh = new TextHelper(x, y, fontsize, text, module.main.getInterpreter(), o, fontFamily);
                o.intrinsicData["Actor"] = sh;

            }, false, false, 'Instanziert ein neues Textobjekt. (x, y) sind die Koordinaten des Textankers (default: links oben), fontsize die Höhe des Textes in Pixeln.', true));

        this.addMethod(new Method("setFontsize", new Parameterlist([
            { identifier: "fontsize", type: doublePrimitiveType, declaration: null, usagePositions: null, isFinal: true },
        ]), null,
            (parameters) => {

                let o: RuntimeObject = parameters[0].value;
                let fontsize: number = parameters[1].value;
                let sh: TextHelper = o.intrinsicData["Actor"];

                sh.setFontsize(fontsize);

            }, false, false, 'Setzt die Schriftgröße des Textes (Einheit: Pixel).', false));

        this.addMethod(new Method("setAlignment", new Parameterlist([
            { identifier: "alignment", type: module.typeStore.getType("Alignment"), declaration: null, usagePositions: null, isFinal: true },
        ]), null,
            (parameters) => {

                let o: RuntimeObject = parameters[0].value;
                let alignment: EnumRuntimeObject = parameters[1].value;
                let sh: TextHelper = o.intrinsicData["Actor"];

                sh.setAlignment(alignment.enumValue.identifier);

            }, false, false, 'Setzt die Ausrichtung in X-Richtung. Zulässige Werte sind "Alignment.left", "Alignment.right" und "Alignment.center".', false));

        this.addMethod(new Method("setText", new Parameterlist([
            { identifier: "text", type: stringPrimitiveType, declaration: null, usagePositions: null, isFinal: true },
        ]), null,
            (parameters) => {

                let o: RuntimeObject = parameters[0].value;
                let text: string = parameters[1].value;
                let sh: TextHelper = o.intrinsicData["Actor"];

                sh.setText(text);

            }, false, false, 'Setzt den Text.', false));

        this.addMethod(new Method("copy", new Parameterlist([
        ]), this,
            (parameters) => {

                let o: RuntimeObject = parameters[0].value;
                let sh: TextHelper = o.intrinsicData["Actor"];

                if (sh.testdestroyed("copy")) return;

                return sh.getCopy(<Klass>o.class);

            }, false, false, 'Erstellt eine Kopie des Text-Objekts und git sie zurück.', false));

        this.addMethod(new Method("getWidth", new Parameterlist([
        ]), doublePrimitiveType,
            (parameters) => {

                let o: RuntimeObject = parameters[0].value;
                let sh: TextHelper = o.intrinsicData["Actor"];

                if (sh.testdestroyed("getWidth")) return;

                return sh.getWidth();

            }, false, false, 'Gibt die Breite des Textes zurück.', false));

        this.addMethod(new Method("getHeight", new Parameterlist([
        ]), doublePrimitiveType,
            (parameters) => {

                let o: RuntimeObject = parameters[0].value;
                let sh: TextHelper = o.intrinsicData["Actor"];

                if (sh.testdestroyed("getHeight")) return;

                return sh.getHeight();

            }, false, false, 'Gibt die Höhe des Textes zurück.', false));


    }

}

export class TextHelper extends FilledShapeHelper {

    alignment: string = "left";

    textStyle: PIXI.TextStyle =
        new PIXI.TextStyle({
            fontFamily: 'Arial',
            fontSize: this.fontsize,
            fontStyle: 'normal',
            fontWeight: 'normal',
            fill: [this.fillColor], // gradient possible...
            stroke: this.borderColor,
            strokeThickness: this.borderWidth,
            dropShadow: false,
            wordWrap: false,
            align: "left"
        });

    constructor(public x: number, public y: number, public fontsize: number,
        public text: string,
        interpreter: Interpreter, runtimeObject: RuntimeObject, public fontFamily?: string) {
        super(interpreter, runtimeObject);
        this.centerXInitial = x;
        this.centerYInitial = y;

        if(this.fontsize == 0) this.fontsize = 10;

        this.borderColor = null;
        this.textStyle.stroke = null;
        if(fontFamily != null){
            this.textStyle.fontFamily = fontFamily;
        }

        this.hitPolygonInitial = [];

        this.render();
        this.addToDefaultGroup();
    }

    getCopy(klass: Klass): RuntimeObject {

        let ro: RuntimeObject = new RuntimeObject(klass);
        let rh: TextHelper = new TextHelper(this.x, this.y, this.fontsize, this.text, this.worldHelper.interpreter, ro);
        ro.intrinsicData["Actor"] = rh;

        rh.alignment = this.alignment;

        rh.copyFrom(this);
        rh.render();

        return ro;
    }


    render(): void {

        let g: PIXI.Text = <any>this.displayObject;
        this.textStyle.fill = this.fillColor;
        this.textStyle.stroke = this.borderColor;
        this.textStyle.strokeThickness = this.borderWidth;
        this.textStyle.fontSize = this.fontsize;

        if (this.displayObject == null) {
            g = new PIXI.Text(this.text, this.textStyle);
            this.displayObject = g;
            this.displayObject.localTransform.translate(this.x, this.y);
            //@ts-ignore
            this.displayObject.transform.onChange();
            this.worldHelper.stage.addChild(g);
        } else {
            g.text = this.text;
            g.alpha = this.fillAlpha;
            switch (this.alignment) {
                case "left": g.anchor.x = 0; break;
                case "center": g.anchor.x = 0.5; break;
                case "right": g.anchor.x = 1.0; break;
            }
            g.style = this.textStyle;
        }

        this.centerXInitial = 0;
        this.centerYInitial = 0;
        if(this.text != null){
            let tm = PIXI.TextMetrics.measureText(this.text, this.textStyle);
    
            this.centerXInitial = tm.width / 2;
            this.centerYInitial = tm.height / 2;
        }


    };

    setFontsize(fontsize: number) {
        this.fontsize = fontsize;
        if(this.fontsize == 0) this.fontsize = 10;
        this.render();
    }

    setText(text: string) {
        this.text = text;
        this.render();
    }

    setAlignment(alignment: string) {
        this.alignment = alignment;
        this.render();
    }

    getWidth(): number {
        let g: PIXI.Text = <any>this.displayObject;
        return g.width;
    }

    getHeight(): number {
        let g: PIXI.Text = <any>this.displayObject;
        return g.height;
    }

}
